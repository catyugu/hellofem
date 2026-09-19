// hellofem::app — point expression evaluation (muparser backend)
// SPDX-License-Identifier: MIT

#include "Expression.h"
#include "units.h"

#include <muParser.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace hellofem::app {

    struct Expression::Impl {
        mu::Parser parser;
        double x = 0, y = 0, z = 0, t = 0;
        std::unordered_map<std::string, double*> vars; // user-owned storage
    };

    Expression::Expression()
        : impl_(std::make_unique<Impl>())
    {
        impl_->parser.DefineVar("x", &impl_->x);
        impl_->parser.DefineVar("y", &impl_->y);
        impl_->parser.DefineVar("z", &impl_->z);
        impl_->parser.DefineVar("t", &impl_->t);
    }

    Expression::~Expression() = default;
    Expression::Expression(Expression&&) noexcept = default;
    Expression& Expression::operator=(Expression&&) noexcept = default;

    bool reserved_variable(std::string_view name)
    {
        return name == "x" or name == "y" or name == "z" or name == "t";
    }

    namespace {
        /// Whether `c` may be part of the number a unit literal applies to.
        bool number_char(char c)
        {
            return (c >= '0' and c <= '9') or c == '.' or c == 'e' or c == 'E'
                or c == '+' or c == '-';
        }

        std::string si_literal(std::string_view literal)
        {
            const double value = parse_si(literal);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.17g", value);
            return std::string(buf);
        }

        /// Replace a numeric-with-unit literal (`20[mV]`) with its numeric SI
        /// value — every such literal in the text, so that a value COMSOL
        /// writes as `238[W/(m*K)]` and a product the app composes from two
        /// of them both parse. Anything else — an expression, a bracket that
        /// does not follow a number, or a unit the parser does not know —
        /// stays as it is, for muparser to accept or reject.
        std::string normalize_units(std::string_view text)
        {
            std::string out;
            std::size_t i = 0;
            while (i < text.size()) {
                const std::size_t open = text.find('[', i);
                const std::size_t close = open == std::string_view::npos
                    ? open
                    : text.find(']', open);
                if (close == std::string_view::npos) {
                    out.append(text.substr(i));
                    break;
                }
                std::size_t begin = open;
                while (begin > i and number_char(text[begin - 1]))
                    --begin;
                out.append(text.substr(i, begin - i));
                const std::string_view literal = text.substr(begin, close - begin + 1);
                try {
                    out += begin == open ? std::string(literal) : si_literal(literal);
                }
                catch (const std::exception&) {
                    out += literal;
                }
                i = close + 1;
            }
            return out;
        }
    } // namespace

    void Expression::parse(std::string_view text,
        std::unordered_map<std::string, double*>& vars)
    {
        expr_ = normalize_units(text);
        impl_->vars = vars;
        impl_->parser.ClearVar();
        // Re-bind the built-in coordinates/time.
        impl_->parser.DefineVar("x", &impl_->x);
        impl_->parser.DefineVar("y", &impl_->y);
        impl_->parser.DefineVar("z", &impl_->z);
        impl_->parser.DefineVar("t", &impl_->t);
        for (const auto& [name, ptr] : vars) {
            if (not reserved_variable(name))
                impl_->parser.DefineVar(name, ptr);
        }
        impl_->parser.SetExpr(expr_);
        // Force parse errors to surface now, not at first eval. muparser's
        // own error type derives from nothing the caller catches, so it is
        // translated here: an expression it rejects must name itself, not end
        // the process without a word.
        try {
            impl_->parser.Eval();
        }
        catch (const mu::ParserError& e) {
            throw std::runtime_error(
                "expression '" + expr_ + "': " + e.GetMsg());
        }
    }

    double Expression::eval(double x, double y, double z, double t)
    {
        impl_->x = x;
        impl_->y = y;
        impl_->z = z;
        impl_->t = t;
        return impl_->parser.Eval();
    }

    std::vector<std::string> Expression::variables() const
    {
        std::vector<std::string> out;
        const auto& vars = impl_->parser.GetUsedVar();
        out.reserve(vars.size());
        for (const auto& [name, value] : vars)
            out.push_back(name);
        std::ranges::sort(out);
        return out;
    }

} // namespace hellofem::app
