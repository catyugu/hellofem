// hellofem::app — COMSOL-style unit parsing
// SPDX-License-Identifier: MIT

#include "units.h"

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hellofem::app {
namespace {

    /// Base units with their SI multiplier (dimensionless ones are 1.0).
    const std::unordered_map<std::string, double>& base_units()
    {
        static const std::unordered_map<std::string, double> units = {
            {"GPa", 1e9}, {"MPa", 1e6}, {"kPa", 1e3}, {"Pa", 1.0},
            {"V", 1.0}, {"mV", 1e-3},
            {"W", 1.0}, {"J", 1.0}, {"kJ", 1e3}, {"K", 1.0},
            {"kg", 1.0}, {"g", 1e-3},
            {"m", 1.0}, {"cm", 1e-2}, {"mm", 1e-3},
            {"s", 1.0}, {"S/m", 1.0}, {"1/K", 1.0}, {"1/m", 1.0},
            {"J/(kg*K)", 1.0}, {"J/(kg*K)", 1.0}, {"W/(m*K)", 1.0},
            {"W/m^2/K", 1.0}, {"W/(m^2*K)", 1.0},
            {"kg/m^3", 1.0}, {"kg/m^2", 1.0}, {"kg/m", 1.0},
        };
        return units;
    }

    /// Recursive-descent parser for unit factor strings.
    class UnitParser {
        std::string_view s;
        std::size_t pos = 0;

        void skip() { while (pos < s.size() and std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; }
        bool match(char c)
        {
            skip();
            if (pos < s.size() and s[pos] == c) {
                ++pos;
                return true;
            }
            return false;
        }

        double number()
        {
            skip();
            std::size_t start = pos;
            while (pos < s.size()
                and (std::isdigit(static_cast<unsigned char>(s[pos])) or s[pos] == '.'
                    or s[pos] == '+' or s[pos] == '-' or s[pos] == 'e' or s[pos] == 'E'))
                ++pos;
            return std::stod(std::string(s.substr(start, pos - start)));
        }

        double primary()
        {
            skip();
            if (match('(')) {
                double v = expr();
                if (!match(')'))
                    throw std::runtime_error("parse_unit: missing ')' in '" + std::string(s) + "'");
                return v;
            }
            if (pos < s.size()
                and (std::isdigit(static_cast<unsigned char>(s[pos])) or s[pos] == '.'))
                return number();
            if (pos < s.size() and std::isalpha(static_cast<unsigned char>(s[pos]))) {
                std::size_t start = pos;
                while (pos < s.size() and std::isalpha(static_cast<unsigned char>(s[pos])))
                    ++pos;
                std::string name(s.substr(start, pos - start));
                auto it = base_units().find(name);
                if (it != base_units().end())
                    return it->second;
                throw std::runtime_error(
                    "parse_unit: unknown unit '" + name + "' in '" + std::string(s) + "'");
            }
            return 1.0;
        }

        double term()
        {
            double v = primary();
            skip();
            if (pos < s.size() and s[pos] == '^') {
                ++pos;
                v = std::pow(v, number());
            }
            return v;
        }

        double expr()
        {
            double v = term();
            for (;;) {
                skip();
                if (pos >= s.size())
                    return v;
                if (s[pos] == '*') {
                    ++pos;
                    v *= term();
                }
                else if (s[pos] == '/') {
                    ++pos;
                    v /= term();
                }
                else
                    return v;
            }
        }

    public:
        explicit UnitParser(std::string_view text)
            : s(text)
        {
        }
        double parse()
        {
            double v = expr();
            skip();
            if (pos != s.size())
                throw std::runtime_error(
                    "parse_unit: trailing characters in '" + std::string(s) + "'");
            return v;
        }
    };

} // namespace

double parse_unit(std::string_view unit)
{
    // Whole-string fast path first (exact table hit).
    std::string trimmed(unit);
    while (!trimmed.empty() and std::isspace(static_cast<unsigned char>(trimmed.front())))
        trimmed.erase(trimmed.begin());
    while (!trimmed.empty() and std::isspace(static_cast<unsigned char>(trimmed.back())))
        trimmed.pop_back();
    if (trimmed.empty())
        return 1.0;
    if (auto it = base_units().find(trimmed); it != base_units().end())
        return it->second;
    return UnitParser(trimmed).parse();
}

double parse_si(std::string_view input)
{
    std::string s(input);
    const auto lb = s.find('[');
    if (lb == std::string::npos)
        return std::stod(s); // Plain number.
    if (s.back() != ']')
        throw std::runtime_error("parse_si: unbalanced '[' in '" + s + "'");
    const std::string value = s.substr(0, lb);
    const std::string unit = s.substr(lb + 1, s.size() - lb - 2);
    return std::stod(value) * parse_unit(unit);
}

} // namespace hellofem::app
