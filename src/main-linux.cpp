#include "hlie_app/hlie_app_controller.h"
#include "hlie_base/hlie_config_service.h"

#include <iostream>
#include <unistd.h>

#include <sys/wait.h>
#include <csignal>
#include <cstdio>
#include <ctime>

// SIGCHLD 信号处理器，用于回收子进程
void sigchld_handler(int signo) {
    (void)signo;
    int status;
    pid_t pid;

    // 循环回收所有已经退出的子进程
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if (WIFEXITED(status)) {
            printf("[SIGCHLD] child %d exited with code %d\n", pid, WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("[SIGCHLD] child %d killed by signal %d\n", pid, WTERMSIG(status));
        }
    }
}

int main(const int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (geteuid() != 0) {
        std::cerr << "Error: Not running as root!" << std::endl;
        return 1;
    }


    // 安装 SIGCHLD 处理器
    struct sigaction sa;
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, nullptr);

    const auto& config_service = hm_HLIE::Singleton<hm_HLIE::HLIEConfigService>::instance();
    config_service->set_args(argc, argv);

    hm_HLIE::HLIELinuxAppController appController;
    int ret = appController.initialize();
    if (ret < 0) {
        std::cout << "HLIECpp appController failed to initialize:" << ret << std::endl;
        return ret;
    }

    ret = appController.run();
    std::cout << "HLIECpp appController run, exit code:" << ret << std::endl;
    return ret;
}
