#include "curlwrapper.hpp"

#include <curl/curl.h>
#include <curl/easy.h>

#include <array>
#include <iostream>

namespace Aronda::Trainer
{

namespace Impl
{

    namespace
    {
        static int writer(char* data, size_t size, size_t nmemb, std::string* writerData)
        {
            if(writerData == NULL) return 0;
            writerData->append(data, size * nmemb);
            return size * nmemb;
        }
    }

    class CurlWrapper
    {
    public:
        CurlWrapper(std::string base_url)
            : m_base_url(std::move(base_url))
        {
            curl_global_init(CURL_GLOBAL_ALL);
            m_curl = curl_easy_init();
            if(!m_curl) throw std::runtime_error("Unable to init curl");

            setOption(CURLOPT_ERRORBUFFER, m_error_buffer.data());
            setOption(CURLOPT_WRITEFUNCTION, writer);
            setOption(CURLOPT_WRITEDATA, &m_buffer);
        }

        ~CurlWrapper()
        {
            curl_easy_cleanup(m_curl);
            curl_global_cleanup();
        }

        std::string get(const std::string& action)
        {
            m_buffer.clear();
            setOption(CURLOPT_URL, (m_base_url + action).c_str());
            performCurl();
            return m_buffer;
        }

        std::string post(const std::string& action, const std::string& data)
        {
            m_buffer.clear();
            setOption(CURLOPT_URL, (m_base_url + action).c_str());
            setOption(CURLOPT_POSTFIELDS, data.c_str());
            performCurl();
            return m_buffer;
        }

    private:
        const std::string m_base_url;
        CURL* m_curl;

        std::array<char, CURL_ERROR_SIZE> m_error_buffer;
        std::string m_buffer;

        void performCurl()
        {
            const auto code = curl_easy_perform(m_curl);
            if(code != CURLE_OK)
            {
                std::cerr << "perform failed" << std::endl;
                throw std::runtime_error("Unable to resquest jaronda");
            }
        }

        template <class T> void setOption(CURLoption option, T data)
        {
            const auto code = curl_easy_setopt(m_curl, option, data);
            if(code != CURLE_OK)
            {
                std::cerr << "set option failed (" << option << "): " << m_error_buffer.data() << " | "
                          << curl_easy_strerror(code) << std::endl;
                throw std::runtime_error("Failed to perform curl action");
            }
        }
    };
}

CurlWrapper::CurlWrapper(std::string base_url)
    : m_pimpl(std::make_unique<Impl::CurlWrapper>(std::move(base_url)))
{
}

CurlWrapper::~CurlWrapper() = default;

std::string CurlWrapper::get(const std::string& action)
{
    return m_pimpl->get(action);
}

std::string CurlWrapper::post(const std::string& action, const std::string& data)
{
    return m_pimpl->post(action, data);
}
}