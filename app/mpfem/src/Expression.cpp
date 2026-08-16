// hellofem::app — point expression evaluation (muparser backend)
// SPDX-License-Identifier: MIT

#include "Expression.h"
#include "units.h"

#include <muParser.h>

#include <algorithm>
#include <cctype>
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

    namespace {
        /// Replace a pure numeric-with-unit literal (`20[mV]`) with the
        /// numeric SI value. Leaves other expressions untouched.
        std::string normalize_units(std::string_view text)
        {
            const std::size_t lb = text.find('[');
            if (lb == std::string::npos or text.back() != ']')
                return std::string(text);
            // Numeric part only (digits/dot/e/sign), no operators.
            std::string_view num = text.substr(0, lb);
            if (num.empty())
                return std::string(text);
            for (char c : num)
                if (!std::isdigit(static_cast<unsigned char>(c)) and c != '.'
                    and c != '+' and c != '-' and c != 'e' and c != 'E')
                    return std::string(text);
            try {
                const double v = parse_si(text);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.17g", v);
                return std::string(buf);
            }
            catch (const std::exception&) {
                return std::string(text); // Leave to muparser to reject.
            }
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
            if (name != "x" and name != "y" and name != "z" and name != "t")
                impl_->parser.DefineVar(name, ptr);
        }
        impl_->parser.SetExpr(expr_);
        // Force parse errors to surface now, not at first eval.
        impl_->parser.Eval();
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
