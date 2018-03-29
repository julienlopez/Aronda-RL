#include "curlwrapper.hpp"

#include <curl/curl.h>
#include <curl/easy.h>

#include <array>
#include <cassert>
#include <iostream>

namespace Aronda::Trainer
{

namespace Impl
{

    static int writer(char* data, size_t size, size_t nmemb, std::string* writerData)
    {
        if(writerData == NULL) return 0;

        writerData->append(data, size * nmemb);

        return size * nmemb;
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

            auto code = curl_easy_setopt(m_curl, CURLOPT_ERRORBUFFER, m_error_buffer.data());
            if(code != CURLE_OK)
            {
                std::wcerr << m_error_buffer.data() << std::endl;
                throw std::runtime_error("Failed to set error buffer");
            }

            code = curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, writer);
            if(code != CURLE_OK)
            {
                std::wcerr << m_error_buffer.data() << std::endl;
                throw std::runtime_error("Failed to set writter");
            }

            code = curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &m_buffer);
            if(code != CURLE_OK)
            {
                std::wcerr << m_error_buffer.data() << std::endl;
                throw std::runtime_error("Failed to set write data");
            }
        }

        ~CurlWrapper()
        {
            curl_global_cleanup();
        }

        std::string get(const std::string& action)
        {
            m_buffer.clear();
            auto code = curl_easy_setopt(m_curl, CURLOPT_URL, (m_base_url + action).c_str());
            if(code != CURLE_OK)
            {
                std::wcerr << m_error_buffer.data() << std::endl;
                throw std::runtime_error("Failed to set url");
            }
            code = curl_easy_perform(m_curl);
            if(code != CURLE_OK)
            {
                std::cerr << "failed" << std::endl;
                throw std::runtime_error("Unable to resquest jaronda");
            }

            curl_easy_cleanup(m_curl);
            return m_buffer;
        }

        std::string post(const std::string& action)
        {
            // TODO CurlWrapper::post
            assert(0 && "not impl yes");
            return {};
        }

    private:
        const std::string m_base_url;
        CURL* m_curl;

        std::array<char, CURL_ERROR_SIZE> m_error_buffer;
        std::string m_buffer;
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

std::string CurlWrapper::post(const std::string& action)
{
    return m_pimpl->post(action);
}
}