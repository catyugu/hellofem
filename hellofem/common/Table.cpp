// hellofem::common — Table implementation
// SPDX-License-Identifier: MIT

#include "Table.h"

#include <algorithm>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    std::string to_str(std::variant<std::string, int, double> value)
    {
        if (std::holds_alternative<int>(value))
            return std::to_string(std::get<int>(value));
        else if (std::holds_alternative<double>(value))
            return std::to_string(std::get<double>(value));
        else if (std::holds_alternative<std::string>(value))
            return std::get<std::string>(value);
        else
            throw std::runtime_error("Variant incorrect");
    }
} // namespace

namespace hellofem {

    Table::Table(std::string title, bool right_justify)
        : name(std::move(title)), _right_justify(right_justify)
    {
    }

    void Table::set(std::string_view row, std::string_view col,
        std::variant<std::string, int, double> value)
    {
        if (std::find(_rows.begin(), _rows.end(), row) == _rows.end())
            _rows.emplace_back(row);
        if (std::find(_cols.begin(), _cols.end(), col) == _cols.end())
            _cols.emplace_back(col);
        _values[std::pair<std::string, std::string>(row, col)]
            = std::move(value);
    }

    std::variant<std::string, int, double>
    Table::get(std::string_view row, std::string_view col) const
    {
        std::pair<std::string, std::string> key(row, col);
        auto it = _values.find(key);
        if (it == _values.end()) {
            throw std::runtime_error(std::format(
                R"(Missing table value for entry ("{}", "{}"))", row, col));
        }
        return it->second;
    }

    std::string Table::str() const
    {
        std::string s;
        std::vector<std::vector<std::string>> tvalues;
        std::vector<std::size_t> col_sizes;

        col_sizes.push_back(name.size());
        for (const auto& c : _cols)
            col_sizes.push_back(c.size());
        for (std::size_t i = 0; i < _rows.size(); i++) {
            tvalues.emplace_back();
            col_sizes[0] = std::max(col_sizes[0], _rows[i].size());
            for (std::size_t j = 0; j < _cols.size(); j++) {
                const std::string value = to_str(get(_rows[i], _cols[j]));
                tvalues[i].push_back(value);
                col_sizes[j + 1] = std::max(col_sizes[j + 1], value.size());
            }
        }

        // Stay silent if no data
        if (tvalues.empty())
            return "";

        std::size_t row_size = 2 * col_sizes.size() + 1;
        for (std::size_t j = 0; j < col_sizes.size(); j++)
            row_size += col_sizes[j];

        std::format_to(std::back_inserter(s), "{:<{}}  |", name, col_sizes[0]);
        for (std::size_t j = 0; j < _cols.size(); j++) {
            if (_right_justify)
                std::format_to(std::back_inserter(s), "  {:>{}}", _cols[j],
                    col_sizes[j + 1]);
            else
                std::format_to(std::back_inserter(s), "  {:<{}}", _cols[j],
                    col_sizes[j + 1]);
        }
        s += "\n" + std::string(row_size, '-');
        for (std::size_t i = 0; i < _rows.size(); i++) {
            std::format_to(std::back_inserter(s), "\n{:<{}}  |", _rows[i],
                col_sizes[0]);
            for (std::size_t j = 0; j < _cols.size(); j++) {
                if (_right_justify)
                    std::format_to(std::back_inserter(s), "  {:>{}}",
                        tvalues[i][j], col_sizes[j + 1]);
                else
                    std::format_to(std::back_inserter(s), "  {:<{}}",
                        tvalues[i][j], col_sizes[j + 1]);
            }
        }

        return s;
    }

} // namespace hellofem
