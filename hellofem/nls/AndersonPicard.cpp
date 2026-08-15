// hellofem::nls — explicit instantiations
// SPDX-License-Identifier: MIT

#include "AndersonPicard.h"

namespace hellofem::nls {

    template class AndersonMixer<double>;
    template class AndersonMixer<float>;

    template AndersonResult<double> anderson_picard<double>(
        const std::function<std::pair<la::MatrixCSR<double>,
            la::Vector<double>>(const la::Vector<double>&)>&,
        la::Vector<double>&, const AndersonConfig&);

    template AndersonResult<float> anderson_picard<float>(
        const std::function<std::pair<la::MatrixCSR<float>,
            la::Vector<float>>(const la::Vector<float>&)>&,
        la::Vector<float>&, const AndersonConfig&);

} // namespace hellofem::nls
