#include "LinuxSeccompHardening.hpp"

#if defined(__linux__)
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <format>
#include <vector>
#endif

namespace core::landrun {

#if defined(__linux__)
namespace {

[[nodiscard]] constexpr sock_filter statement(unsigned short code,
                                               unsigned int value) {
    return sock_filter{code, 0, 0, value};
}

[[nodiscard]] constexpr sock_filter jump(unsigned short code,
                                          unsigned int value,
                                          unsigned char on_true,
                                          unsigned char on_false) {
    return sock_filter{code, on_true, on_false, value};
}

void deny_syscall(std::vector<sock_filter>& filter, int syscall_number) {
    filter.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K,
                          static_cast<unsigned int>(syscall_number), 0, 1));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM));
}

} // namespace
#endif

LandrunResult apply_linux_seccomp_hardening(bool deny_network) {
#if !defined(__linux__)
    (void)deny_network;
    return {.success = true};
#else
#if defined(__x86_64__)
    constexpr unsigned int expected_arch = AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
    constexpr unsigned int expected_arch = AUDIT_ARCH_AARCH64;
#else
    return {.success = false,
            .detail = "seccomp hardening supports Linux x86_64 and aarch64 only"};
#endif

    std::vector<sock_filter> filter;
    filter.reserve(96);
    filter.push_back(statement(
        BPF_LD | BPF_W | BPF_ABS,
        static_cast<unsigned int>(offsetof(seccomp_data, arch))));
    filter.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K, expected_arch, 1, 0));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    filter.push_back(statement(
        BPF_LD | BPF_W | BPF_ABS,
        static_cast<unsigned int>(offsetof(seccomp_data, nr))));

#if defined(__x86_64__) && defined(__X32_SYSCALL_BIT)
    // x32 shares AUDIT_ARCH_X86_64 but tags syscall numbers with bit 30.
    // Reject that alternate ABI so it cannot bypass the native denylist.
    filter.push_back(jump(BPF_JMP | BPF_JSET | BPF_K,
                          __X32_SYSCALL_BIT, 0, 1));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM));
#endif

#ifdef __NR_clone
    constexpr unsigned int namespace_clone_flags =
        CLONE_NEWCGROUP | CLONE_NEWIPC | CLONE_NEWNET | CLONE_NEWNS
        | CLONE_NEWPID | CLONE_NEWUSER | CLONE_NEWUTS
#ifdef CLONE_NEWTIME
        | CLONE_NEWTIME
#endif
        ;
    // Keep ordinary process/thread creation while preventing clone(2) from
    // creating namespaces. Landlock restrictions still inherit across clone.
    filter.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K,
                          static_cast<unsigned int>(__NR_clone), 0, 3));
    filter.push_back(statement(
        BPF_LD | BPF_W | BPF_ABS,
        static_cast<unsigned int>(offsetof(seccomp_data, args))));
    filter.push_back(jump(BPF_JMP | BPF_JSET | BPF_K,
                          namespace_clone_flags, 0, 1));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM));
    filter.push_back(statement(
        BPF_LD | BPF_W | BPF_ABS,
        static_cast<unsigned int>(offsetof(seccomp_data, nr))));
#endif

#ifdef __NR_ptrace
    deny_syscall(filter, __NR_ptrace);
#endif
#ifdef __NR_process_vm_readv
    deny_syscall(filter, __NR_process_vm_readv);
#endif
#ifdef __NR_process_vm_writev
    deny_syscall(filter, __NR_process_vm_writev);
#endif
#ifdef __NR_mount
    deny_syscall(filter, __NR_mount);
#endif
#ifdef __NR_umount2
    deny_syscall(filter, __NR_umount2);
#endif
#ifdef __NR_pivot_root
    deny_syscall(filter, __NR_pivot_root);
#endif
#ifdef __NR_move_mount
    deny_syscall(filter, __NR_move_mount);
#endif
#ifdef __NR_open_tree
    deny_syscall(filter, __NR_open_tree);
#endif
#ifdef __NR_fsopen
    deny_syscall(filter, __NR_fsopen);
#endif
#ifdef __NR_fsmount
    deny_syscall(filter, __NR_fsmount);
#endif
#ifdef __NR_mount_setattr
    deny_syscall(filter, __NR_mount_setattr);
#endif
#ifdef __NR_bpf
    deny_syscall(filter, __NR_bpf);
#endif
#ifdef __NR_perf_event_open
    deny_syscall(filter, __NR_perf_event_open);
#endif
#ifdef __NR_userfaultfd
    deny_syscall(filter, __NR_userfaultfd);
#endif
#ifdef __NR_io_uring_setup
    deny_syscall(filter, __NR_io_uring_setup);
#endif
#ifdef __NR_io_uring_enter
    deny_syscall(filter, __NR_io_uring_enter);
#endif
#ifdef __NR_io_uring_register
    deny_syscall(filter, __NR_io_uring_register);
#endif
#ifdef __NR_kexec_load
    deny_syscall(filter, __NR_kexec_load);
#endif
#ifdef __NR_init_module
    deny_syscall(filter, __NR_init_module);
#endif
#ifdef __NR_finit_module
    deny_syscall(filter, __NR_finit_module);
#endif
#ifdef __NR_delete_module
    deny_syscall(filter, __NR_delete_module);
#endif
#ifdef __NR_unshare
    deny_syscall(filter, __NR_unshare);
#endif
#ifdef __NR_setns
    deny_syscall(filter, __NR_setns);
#endif

    if (deny_network) {
#ifdef __NR_socket
        deny_syscall(filter, __NR_socket);
#endif
#ifdef __NR_connect
        deny_syscall(filter, __NR_connect);
#endif
#ifdef __NR_bind
        deny_syscall(filter, __NR_bind);
#endif
#ifdef __NR_listen
        deny_syscall(filter, __NR_listen);
#endif
#ifdef __NR_accept
        deny_syscall(filter, __NR_accept);
#endif
#ifdef __NR_accept4
        deny_syscall(filter, __NR_accept4);
#endif
        // Keep anonymous AF_UNIX socket pairs and read/write IPC usable.
        // With socket/connect/bind/listen denied and inherited FDs closed,
        // those operations cannot establish an external network channel.
    }

    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    sock_fprog program{
        .len = static_cast<unsigned short>(filter.size()),
        .filter = filter.data(),
    };
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        return {.success = false,
                .detail = std::format("PR_SET_NO_NEW_PRIVS failed: {}",
                                      std::strerror(errno))};
    }
    if (::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
        return {.success = false,
                .detail = std::format("seccomp filter installation failed: {}",
                                      std::strerror(errno))};
    }
    return {.success = true};
#endif
}

} // namespace core::landrun
