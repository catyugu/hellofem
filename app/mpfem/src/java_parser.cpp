// hellofem::app — parse a clean COMSOL Java model script into ModelScript
// SPDX-License-Identifier: MIT

#include "java_parser.h"

#include "property.h"
#include "units.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hellofem::app {
    namespace {

        // ---------------------------------------------------------------------
        // Tokenizer
        // ---------------------------------------------------------------------

        enum class TokKind { ident,
            string,
            number,
            punct,
            eof };

        struct Token {
            TokKind kind;
            std::string text; // identifier, number, punctuation char, or decoded string
            int line;
        };

        /// Decode a Java string literal body (handles \\ " \n \t etc.).
        std::string decode_string(const std::string& body)
        {
            std::string out;
            out.reserve(body.size());
            for (std::size_t i = 0; i < body.size(); ++i) {
                if (body[i] == '\\' and i + 1 < body.size()) {
                    const char c = body[++i];
                    switch (c) {
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '"':
                        out += '"';
                        break;
                    case '\'':
                        out += '\'';
                        break;
                    default:
                        out += c;
                        break;
                    }
                }
                else {
                    out += body[i];
                }
            }
            return out;
        }

        std::vector<Token> tokenize(const std::string& src)
        {
            std::vector<Token> tokens;
            std::size_t i = 0;
            int line = 1;
            while (i < src.size()) {
                const char c = src[i];
                if (c == '\n') {
                    ++line;
                    ++i;
                    continue;
                }
                if (std::isspace(static_cast<unsigned char>(c))) {
                    ++i;
                    continue;
                }
                if (c == '/' and i + 1 < src.size() and src[i + 1] == '/') {
                    while (i < src.size() and src[i] != '\n')
                        ++i;
                    continue;
                }
                if (c == '/' and i + 1 < src.size() and src[i + 1] == '*') {
                    i += 2;
                    while (i + 1 < src.size() and not(src[i] == '*' and src[i + 1] == '/')) {
                        if (src[i] == '\n')
                            ++line;
                        ++i;
                    }
                    i += 2;
                    continue;
                }
                if (c == '"') {
                    std::string body;
                    ++i;
                    while (i < src.size() and src[i] != '"') {
                        if (src[i] == '\\' and i + 1 < src.size())
                            body += src[i++], body += src[i++];
                        else
                            body += src[i++];
                    }
                    if (i >= src.size())
                        throw std::runtime_error("java: unterminated string at line " + std::to_string(line));
                    ++i; // closing quote
                    tokens.push_back({TokKind::string, decode_string(body), line});
                    continue;
                }
                if (std::isalpha(static_cast<unsigned char>(c)) or c == '_' or c == '$') {
                    std::size_t start = i;
                    while (i < src.size()
                        and (std::isalnum(static_cast<unsigned char>(src[i])) or src[i] == '_'
                            or src[i] == '$'))
                        ++i;
                    tokens.push_back({TokKind::ident, src.substr(start, i - start), line});
                    continue;
                }
                if (std::isdigit(static_cast<unsigned char>(c))
                    or (c == '.' and i + 1 < src.size()
                        and std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
                    std::size_t start = i;
                    while (i < src.size()
                        and (std::isdigit(static_cast<unsigned char>(src[i])) or src[i] == '.'
                            or src[i] == 'e' or src[i] == 'E' or src[i] == '+' or src[i] == '-'))
                        ++i;
                    tokens.push_back({TokKind::number, src.substr(start, i - start), line});
                    continue;
                }
                tokens.push_back({TokKind::punct, std::string(1, c), line});
                ++i;
            }
            tokens.push_back({TokKind::eof, "", line});
            return tokens;
        }

        // ---------------------------------------------------------------------
        // Statement chain parsing: `model.param().set(...) ;`
        // ---------------------------------------------------------------------

        struct Call {
            std::string method;
            std::vector<std::vector<Token>> args; // each arg = token slice
        };

        struct Chain {
            std::vector<Call> calls;
        };

        class Parser {
            std::vector<Token> toks;
            std::size_t pos = 0;

            const Token& peek() const { return toks[std::min(pos, toks.size() - 1)]; }
            Token next()
            {
                const Token t = peek();
                if (t.kind != TokKind::eof)
                    ++pos;
                return t;
            }
            [[noreturn]] void err(const std::string& msg) const
            {
                throw std::runtime_error("java: " + msg + " at line "
                    + std::to_string(peek().line));
            }

            /// Split tokens at top-level commas (depth 0) of `[begin,end)`.
            std::vector<std::vector<Token>> split_args(std::size_t begin, std::size_t end) const
            {
                std::vector<std::vector<Token>> args;
                std::vector<Token> cur;
                int depth = 0;
                for (std::size_t i = begin; i < end; ++i) {
                    const Token& t = toks[i];
                    if (t.kind == TokKind::punct) {
                        if (t.text == "(" or t.text == "[" or t.text == "{")
                            ++depth;
                        else if (t.text == ")" or t.text == "]" or t.text == "}")
                            --depth;
                        else if (t.text == "," and depth == 0) {
                            args.push_back(std::move(cur));
                            cur.clear();
                            continue;
                        }
                    }
                    cur.push_back(t);
                }
                if (not cur.empty())
                    args.push_back(std::move(cur));
                return args;
            }

        public:
            explicit Parser(std::vector<Token> tokens)
                : toks(std::move(tokens))
            {
            }

            /// Parse all top-level statements. Handles `new` declarations and
            /// plain expressions by skipping to `;`.
            std::vector<Chain> parse_all()
            {
                std::vector<Chain> chains;
                while (peek().kind != TokKind::eof) {
                    if (peek().kind == TokKind::punct and peek().text == ";") {
                        next();
                        continue;
                    }
                    // Skip a leading declaration keyword (`import`, `public`,
                    // `class`, `static`, `void`, `int`, `double`, `String`).
                    while (peek().kind == TokKind::ident
                        and (peek().text == "import" or peek().text == "public"
                            or peek().text == "class" or peek().text == "static"
                            or peek().text == "void" or peek().text == "final"
                            or peek().text == "int" or peek().text == "double"
                            or peek().text == "String" or peek().text == "boolean")) {
                        if (peek().text == "import") {
                            // Skip to the `;`.
                            while (peek().kind != TokKind::eof and not(peek().kind == TokKind::punct and peek().text == ";"))
                                next();
                            next(); // ;
                            continue;
                        }
                        next();
                    }
                    Chain chain;
                    // Must start with `model` (possibly `Model model = ModelUtil...`).
                    if (peek().kind == TokKind::ident and peek().text == "model"
                        and pos + 1 < toks.size() and toks[pos + 1].kind == TokKind::punct
                        and toks[pos + 1].text == ".") {
                        while (peek().kind != TokKind::eof and not(peek().kind == TokKind::punct and peek().text == ";")) {
                            if (peek().kind == TokKind::punct and peek().text == ".")
                                next();
                            if (peek().kind != TokKind::ident)
                                break;
                            Call call;
                            call.method = next().text;
                            if (peek().kind == TokKind::punct and peek().text == "(") {
                                next();
                                std::size_t open = pos - 1;
                                int depth = 1;
                                while (depth > 0) {
                                    if (peek().kind == TokKind::eof)
                                        err("unterminated call '" + call.method + "'");
                                    if (peek().kind == TokKind::punct and peek().text == "(")
                                        ++depth;
                                    else if (peek().kind == TokKind::punct and peek().text == ")")
                                        --depth;
                                    next();
                                }
                                std::size_t close = pos - 1; // position of ')'
                                call.args = split_args(open + 1, close);
                            }
                            chain.calls.push_back(std::move(call));
                        }
                        // The chain starts with the receiver `model`; drop it so
                        // `param().set(...)` reads as calls[0]=param, calls[1]=set.
                        if (not chain.calls.empty() and chain.calls[0].method == "model")
                            chain.calls.erase(chain.calls.begin());
                        if (not chain.calls.empty())
                            chains.push_back(std::move(chain));
                    }
                    // Skip to the `;`.
                    while (peek().kind != TokKind::eof and not(peek().kind == TokKind::punct and peek().text == ";"))
                        next();
                    if (peek().kind == TokKind::punct and peek().text == ";")
                        next();
                }
                return chains;
            }
        };

        // ---------------------------------------------------------------------
        // Chain interpretation
        // ---------------------------------------------------------------------

        /// Join an argument's tokens into one string (strings unquoted,
        /// punctuation/spaces stripped so `9[cm]` stays `9[cm]`).
        std::string arg_string(const std::vector<Token>& arg)
        {
            std::string out;
            for (const Token& t : arg)
                out += t.text;
            return out;
        }

        /// Join only the string tokens of an argument (space-separated). Used
        /// for property values, where a `new String[]{...}` array becomes a
        /// space-separated list (e.g. a 9-entry tensor).
        std::string arg_strings(const std::vector<Token>& arg)
        {
            std::string out;
            for (const Token& t : arg)
                if (t.kind == TokKind::string) {
                    if (!out.empty())
                        out += ' ';
                    out += t.text;
                }
            return out;
        }

        /// Property/feature value: the space-joined string tokens if any,
        /// else the raw token text (for numeric values).
        std::string arg_value(const std::vector<Token>& arg)
        {
            const std::string s = arg_strings(arg);
            return s.empty() ? arg_string(arg) : s;
        }

        /// Parse a `set(...)` selection: a `new int[]{...}` / `new String[]{...}`
        /// array, or several arguments (`set(2, 3, 4, 5)`), into a set of ids.
        std::set<int> parse_selection(const std::vector<std::vector<Token>>& args)
        {
            std::set<int> ids;
            for (const std::vector<Token>& arg : args)
                for (const Token& t : arg) {
                    if (t.kind == TokKind::number)
                        ids.insert(static_cast<int>(std::stod(t.text)));
                    else if (t.kind == TokKind::string) {
                        try {
                            ids.insert(static_cast<int>(std::stod(t.text)));
                        }
                        catch (...) { /* non-numeric string: ignore */
                        }
                    }
                }
            return ids;
        }

        /// The element of `items` tagged `tag`, or nullptr.
        template <class Item>
        Item* find_tagged(std::vector<Item>& items, std::string_view tag)
        {
            for (Item& item : items)
                if (item.tag == tag)
                    return &item;
            return nullptr;
        }

        /// Split a comma-separated argument list.
        std::vector<std::string> split_commas(const std::string& text)
        {
            std::vector<std::string> parts;
            std::string current;
            for (char c : text) {
                if (c == ',') {
                    parts.push_back(current);
                    current.clear();
                }
                else
                    current += c;
            }
            if (not current.empty())
                parts.push_back(current);
            return parts;
        }

        /// Evaluate a numeric model expression (a parameter reference or a
        /// literal with an optional unit) at the model parameters.
        double eval_value(const std::string& text,
            const std::unordered_map<std::string, double>& params)
        {
            if (text.empty())
                throw std::runtime_error("java: empty value");
            return ScalarExpression(text, params).eval(0, 0, 0, 0);
        }

        /// Resolve a time list into its levels: `range(t0, dt, t1)` (a
        /// constant step over a closed interval, as COMSOL's range) or an
        /// explicit comma-separated list of times. Each entry is a model
        /// expression, so it may name parameters.
        std::vector<double> resolve_time_list(const std::string& text,
            const std::unordered_map<std::string, double>& params)
        {
            std::vector<double> times;
            const std::size_t open = text.find('(');
            if (text.starts_with("range") and open != std::string::npos
                and text.back() == ')') {
                auto parts = split_commas(text.substr(open + 1, text.size() - open - 2));
                if (parts.size() != 3)
                    throw std::runtime_error("java: range() needs three arguments");
                const double t0 = eval_value(parts[0], params);
                const double dt = eval_value(parts[1], params);
                const double t1 = eval_value(parts[2], params);
                const double count = std::floor((t1 - t0) / dt + 1e-9);
                if (count < 1.0 or count > 1e6)
                    throw std::runtime_error("java: invalid time list '" + text + "'");
                for (int i = 0; i <= static_cast<int>(count); ++i)
                    times.push_back(t0 + i * dt);
                return times;
            }
            for (const std::string& part : split_commas(text))
                if (not part.empty())
                    times.push_back(eval_value(part, params));
            return times;
        }

        /// `model.param().set(name, value, [desc])`.
        void interpret_parameter(const std::vector<Call>& c, ModelScript& model)
        {
            if (c.size() < 2 or c[1].method != "set" or c[1].args.size() < 2)
                return;
            Parameter p;
            p.name = arg_string(c[1].args[0]);
            p.value = arg_string(c[1].args[1]);
            try {
                p.si = parse_si(p.value);
            }
            catch (const std::exception&) {
                p.si = 0.0; // expression referencing other params
            }
            model.parameters.push_back(std::move(p));
            return;
        }

        /// `model.component(...)`: the materials, the physics interfaces and
        /// the multiphysics couplings of one component, by their tags.
        void interpret_component(const std::vector<Call>& c, ModelScript& model)
        {
            if (c.size() < 2 or c[0].args.empty()
                or arg_string(c[0].args[0]).empty())
                return;
            // material().create / material(tag).X
            if (c.size() >= 2 and c[1].method == "material") {
                if (c[1].args.empty() and c.size() >= 3 and c[2].method == "create") {
                    Material m;
                    m.tag = arg_string(c[2].args[0]);
                    model.materials.push_back(std::move(m));
                    return;
                }
                const std::string mtag = c[1].args.empty() ? "" : arg_string(c[1].args[0]);
                auto* mat = find_tagged(model.materials, mtag);
                if (mat == nullptr)
                    return;
                // propertyGroup(pg).set(prop, value) or materialModel().create then propertyGroup
                std::size_t k = 2;
                while (k + 1 < c.size()) {
                    if (c[k].method == "propertyGroup" and k + 1 < c.size() and c[k + 1].method == "set") {
                        const std::string prop = arg_string(c[k + 1].args[0]);
                        std::string value = arg_value(c[k + 1].args[1]);
                        mat->properties.push_back({prop, value});
                        return;
                    }
                    if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                        mat->domains = parse_selection(c[k + 1].args);
                        return;
                    }
                    if (c[k].method == "materialModel" and k + 1 < c.size() and c[k + 1].method == "create")
                        k += 2; // skip materialModel().create
                    else
                        return;
                }
                return;
            }

            // physics().create / physics(tag).create / physics(tag).feature(feat).set
            if (c.size() >= 2 and c[1].method == "physics") {
                if (c[1].args.empty() and c.size() >= 3 and c[2].method == "create") {
                    Physics ph;
                    ph.tag = arg_string(c[2].args[0]);
                    ph.type = arg_string(c[2].args[1]);
                    model.physics.push_back(std::move(ph));
                    return;
                }
                const std::string ptag = c[1].args.empty() ? "" : arg_string(c[1].args[0]);
                auto* ph = find_tagged(model.physics, ptag);
                if (ph == nullptr)
                    return;
                // physics(tag).prop("ShapeProperty").set("order_<var>", "N"):
                // the element order of one of the interface's dependent
                // variables. COMSOL writes `2s` for its quadratic serendipity
                // form, whose leading digit is the order the app's Lagrange
                // basis answers for.
                if (c.size() >= 4 and c[2].method == "prop"
                    and c[3].method == "set"
                    and arg_string(c[2].args[0]) == "ShapeProperty") {
                    const std::string key = arg_string(c[3].args[0]);
                    if (c[3].args.size() > 1 and key.starts_with("order_"))
                        ph->element_order[key.substr(6)]
                            = std::stoi(arg_string(c[3].args[1]));
                    return;
                }
                // physics(tag).create(feat, type, dim)
                if (c.size() >= 3 and c[2].method == "create") {
                    PhysicsFeature f;
                    f.tag = arg_string(c[2].args[0]);
                    f.type = arg_string(c[2].args[1]);
                    ph->features.push_back(std::move(f));
                    return;
                }
                // physics(tag).feature(feat).set / .selection().set
                if (c.size() >= 3 and c[2].method == "feature") {
                    const std::string ftag = arg_string(c[2].args[0]);
                    auto* feat = find_tagged(ph->features, ftag);
                    if (feat == nullptr) {
                        // A `.feature(tag).set()` without a matching create
                        // (e.g. referencing a default feature) — track the tag.
                        ph->features.push_back({ftag, ftag, {}, {}});
                        feat = &ph->features.back();
                    }
                    std::size_t k = 3;
                    while (k < c.size()) {
                        if (c[k].method == "set") {
                            const std::string key = arg_string(c[k].args[0]);
                            const std::string value = c[k].args.size() > 1 ? arg_value(c[k].args[1]) : "";
                            feat->properties[key] = value;
                            return;
                        }
                        if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                            feat->selection = parse_selection(c[k + 1].args);
                            return;
                        }
                        ++k;
                    }
                }
                return;
            }

            // multiphysics().create / multiphysics(tag).set / .selection().set
            if (c.size() >= 2 and c[1].method == "multiphysics") {
                if (c[1].args.empty() and c.size() >= 3 and c[2].method == "create") {
                    MultiphysicsCoupling mc;
                    mc.tag = arg_string(c[2].args[0]);
                    mc.type = arg_string(c[2].args[1]);
                    model.couplings.push_back(std::move(mc));
                    return;
                }
                const std::string ctag = c[1].args.empty() ? "" : arg_string(c[1].args[0]);
                auto* mc = find_tagged(model.couplings, ctag);
                if (mc == nullptr)
                    return;
                std::size_t k = 2;
                while (k < c.size()) {
                    if (c[k].method == "set") {
                        const std::string key = arg_string(c[k].args[0]);
                        const std::string value = c[k].args.size() > 1 ? arg_value(c[k].args[1]) : "";
                        mc->properties[key] = value;
                        return;
                    }
                    if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                        mc->domains = parse_selection(c[k + 1].args);
                        return;
                    }
                    ++k;
                }
                return;
            }
        }

        /// `model.study(...)`: the study step type (a stationary or a
        /// transient driver) and the time list of a transient one.
        void interpret_study(const std::vector<Call>& c, ModelScript& model)
        {
            if (c.size() < 2)
                return;
            // study(tag).create(step, type): the study step, whose type
            // selects the stationary or the transient driver.
            if (c[1].method == "create") {
                if (c[1].args.size() < 2)
                    return; // study().create(tag): the study itself
                const std::string step_type = arg_string(c[1].args[1]);
                if (step_type == "Stationary")
                    model.study.transient = false;
                else if (step_type == "Transient")
                    model.study.transient = true;
                return;
            }
            if (c.size() >= 3 and c[1].method == "feature" and c[2].method == "set") {
                const std::string key = arg_string(c[2].args[0]);
                const std::string value = c[2].args.size() > 1 ? arg_string(c[2].args[1]) : "";
                if (key == "tlist")
                    model.study.times_expr = value;
                return;
            }
            return;
        }

        /// `model.result(...).export(...)`: the expressions a Data export
        /// writes, one column per expression.
        void interpret_result(const std::vector<Call>& c, ModelScript& model)
        {
            if (c.size() >= 3 and c[1].method == "export" and c[2].method == "create")
                return; // export feature creation
            if (c.size() >= 3 and c[1].method == "export" and c[2].method == "set") {
                const std::string key = arg_string(c[2].args[0]);
                if (key == "expr") {
                    model.export_config.expressions.clear();
                    if (c[2].args.size() > 1) {
                        for (const Token& t : c[2].args[1]) {
                            if (t.kind == TokKind::string)
                                model.export_config.expressions.push_back(t.text);
                        }
                    }
                }
                return;
            }
            return;
        }

        /// Interpret one chain and update `model`.
        void interpret(const Chain& chain, ModelScript& model)
        {
            const auto& c = chain.calls;
            if (c.empty())
                return;

            if (c[0].method == "param")
                interpret_parameter(c, model);
            else if (c[0].method == "component")
                interpret_component(c, model);
            else if (c[0].method == "study")
                interpret_study(c, model);
            else if (c[0].method == "result")
                interpret_result(c, model);
            // `model.save(...)`, `model.modelPath(...)` — no effect.
        }

    } // namespace

    ModelScript parse_model_java(const std::filesystem::path& filename)
    {
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("parse_model_java: cannot open '" + filename.string() + "'");
        std::ostringstream ss;
        ss << file.rdbuf();

        Parser parser(tokenize(ss.str()));
        auto chains = parser.parse_all();
        ModelScript model;
        model.name = filename.stem().string();
        for (const auto& chain : chains)
            interpret(chain, model);
        // The time list may reference parameters, so it resolves once the
        // whole script has been read.
        if (not model.study.times_expr.empty()) {
            std::unordered_map<std::string, double> params;
            for (const Parameter& p : model.parameters)
                params[p.name] = p.si;
            model.study.times = resolve_time_list(model.study.times_expr, params);
        }
        return model;
    }

} // namespace hellofem::app
