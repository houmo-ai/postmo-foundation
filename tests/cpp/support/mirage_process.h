#ifndef HLIE_TESTS_MIRAGE_PROCESS_H
#define HLIE_TESTS_MIRAGE_PROCESS_H

#include <memory>
#include <string>

class MirageProcess {
public:
    MirageProcess();
    ~MirageProcess();

    MirageProcess(const MirageProcess&) = delete;
    MirageProcess& operator=(const MirageProcess&) = delete;
    MirageProcess(MirageProcess&&) = delete;
    MirageProcess& operator=(MirageProcess&&) = delete;

    [[nodiscard]] const std::string& base_url() const noexcept { return base_url_; }
    [[nodiscard]] const std::string& control_url() const noexcept { return control_url_; }
    [[nodiscard]] const std::string& host() const noexcept { return host_; }
    [[nodiscard]] int port() const noexcept { return port_; }

private:
    struct Impl;

    void launch(const std::string& executable);
    void wait_until_ready();
    void stop() noexcept;

    std::unique_ptr<Impl> impl_;
    std::string ready_file_;
    std::string base_url_;
    std::string control_url_;
    std::string host_;
    int port_ = 0;
};

#endif // HLIE_TESTS_MIRAGE_PROCESS_H
