// hellofem::app — parse a clean COMSOL Java model script into ModelScript
// SPDX-License-Identifier: MIT

#include "java_parser.h"
#include "units.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hellofem::app {
namespace {

    // ---------------------------------------------------------------------
    // Tokenizer
    // ---------------------------------------------------------------------

    enum class TokKind { ident, string, number, punct, eof };

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
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '\\': out += '\\'; break;
                case '"': out += '"'; break;
                case '\'': out += '\''; break;
                default: out += c; break;
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
                while (i + 1 < src.size() and not (src[i] == '*' and src[i + 1] == '/')) {
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
                        while (peek().kind != TokKind::eof and not (peek().kind == TokKind::punct and peek().text == ";"))
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
                    while (peek().kind != TokKind::eof and not (peek().kind == TokKind::punct and peek().text == ";")) {
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
                while (peek().kind != TokKind::eof and not (peek().kind == TokKind::punct and peek().text == ";"))
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

    /// Parse a `new int[]{...}` / `new String[]{...}` selection arg into a
    /// set of integers (single int args and `new int[]{}` supported).
    std::set<int> parse_selection(const std::vector<Token>& arg)
    {
        std::set<int> ids;
        for (const Token& t : arg) {
            if (t.kind == TokKind::number)
                ids.insert(static_cast<int>(std::stod(t.text)));
            else if (t.kind == TokKind::string) {
                try {
                    ids.insert(static_cast<int>(std::stod(t.text)));
                }
                catch (...) { /* non-numeric string: ignore */ }
            }
        }
        return ids;
    }

    /// Interpret one chain and update `model`.
    void interpret(const Chain& chain, ModelScript& model)
    {
        const auto& c = chain.calls;
        if (c.empty())
            return;

        // `model.param().set(name, value, [desc])` -> calls [param, set].
        if (c.size() >= 2 and c[0].method == "param" and c[1].method == "set"
            and c[1].args.size() >= 2) {
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

        // `model.component().create(name, true)` — create component.
        if (c[0].method == "component" and c[0].args.empty() and c.size() >= 2
            and c[1].method == "create")
            return; // component creation: nothing to record

        // Material creation: `model.component(comp).material().create(tag, type)`
        // Material property: `... .material(tag).propertyGroup(pg).set(prop, value)`
        // Material selection: `... .material(tag).selection().set(ids)`
        // Material model: `... .material(tag).materialModel().create(mtag, type)`
        if (c[0].method == "component" and c[0].args.size() == 1) {
            const std::string comp = arg_string(c[0].args[0]);
            if (comp.empty())
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
                auto* mat = [&]() -> Material* {
                    for (auto& m : model.materials)
                        if (m.tag == mtag)
                            return &m;
                    return nullptr;
                }();
                if (mat == nullptr)
                    return;
                // propertyGroup(pg).set(prop, value) or materialModel().create then propertyGroup
                std::size_t k = 2;
                while (k + 1 < c.size()) {
                    if (c[k].method == "propertyGroup" and k + 1 < c.size() and c[k + 1].method == "set") {
                        const std::string prop = arg_string(c[k + 1].args[0]);
                        std::string value = arg_string(c[k + 1].args[1]);
                        mat->properties.push_back({prop, value});
                        return;
                    }
                    if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                        mat->domains = parse_selection(c[k + 1].args[0]);
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
                auto* ph = [&]() -> Physics* {
                    for (auto& p : model.physics)
                        if (p.tag == ptag)
                            return &p;
                    return nullptr;
                }();
                if (ph == nullptr)
                    return;
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
                    auto* feat = [&]() -> PhysicsFeature* {
                        for (auto& f : ph->features)
                            if (f.tag == ftag)
                                return &f;
                        return nullptr;
                    }();
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
                            const std::string value = c[k].args.size() > 1 ? arg_string(c[k].args[1]) : "";
                            feat->properties[key] = value;
                            return;
                        }
                        if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                            feat->selection = parse_selection(c[k + 1].args[0]);
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
                auto* mc = [&]() -> MultiphysicsCoupling* {
                    for (auto& m : model.couplings)
                        if (m.tag == ctag)
                            return &m;
                    return nullptr;
                }();
                if (mc == nullptr)
                    return;
                std::size_t k = 2;
                while (k < c.size()) {
                    if (c[k].method == "set") {
                        const std::string key = arg_string(c[k].args[0]);
                        const std::string value = c[k].args.size() > 1 ? arg_string(c[k].args[1]) : "";
                        mc->properties[key] = value;
                        return;
                    }
                    if (c[k].method == "selection" and k + 1 < c.size() and c[k + 1].method == "set") {
                        mc->domains = parse_selection(c[k + 1].args[0]);
                        return;
                    }
                    ++k;
                }
                return;
            }

            // mesh().create / mesh(tag).autoMeshSize / mesh(tag).run
            if (c.size() >= 2 and c[1].method == "mesh") {
                if (c.size() >= 3 and c[2].method == "autoMeshSize") {
                    try {
                        model.study.mesh_refine = static_cast<int>(std::stod(arg_string(c[2].args[0])));
                    }
                    catch (...) {}
                }
                return;
            }
        }

        // study().create / study(tag).create / study(tag).feature(feat).set / study(tag).run
        if (c[0].method == "study") {
            if (c[0].args.empty() and c.size() >= 3 and c[1].method == "create")
                return; // study creation
            const std::string stag = c[0].args.empty() ? "" : arg_string(c[0].args[0]);
            if (c.size() >= 3 and c[1].method == "create") {
                const std::string feat_type = arg_string(c[2].args[1]);
                if (feat_type == "Stationary")
                    model.study.type = "Stationary";
                else if (feat_type == "Transient")
                    model.study.type = "Transient";
                return;
            }
            if (c.size() >= 3 and c[1].method == "feature" and c[2].method == "set") {
                const std::string key = arg_string(c[2].args[0]);
                const std::string value = c[2].args.size() > 1 ? arg_string(c[2].args[1]) : "";
                if (key == "tlist")
                    model.study.times = {}; // parsed from the range(...) expression
                return;
            }
            return;
        }

        // result().export().create / result().export(tag).set / .run
        if (c[0].method == "result") {
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
    return model;
}

} // namespace hellofem::app
