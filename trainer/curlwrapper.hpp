#pragma once

#include <memory>
#include <string>

namespace Aronda::Trainer
{

namespace Impl
{
    class CurlWrapper;
}

class CurlWrapper
{
public:
    CurlWrapper(std::string base_url);

    ~CurlWrapper();

    std::string get(const std::string& action);

    std::string post(const std::string& action);

private:
    std::unique_ptr<Impl::CurlWrapper> m_pimpl;
};
}
