#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>

// 定义日志级别
enum class LogLevel
{
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// 将日志级别转换为字符串
const char *logLevelToString(LogLevel level);

// 获取当前时间的字符串表示
std::string getCurrentTime();

// 日志类
class Logger
{
public:
    // 构造函数
    Logger(const std::string &filename, LogLevel minLevel = LogLevel::INFO) : minLogLevel(minLevel)
    {
        logFile.open(filename, std::ios::out | std::ios::app);
        if (!logFile.is_open())
        {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    // 析构函数
    ~Logger()
    {
        if (logFile.is_open())
        {
            logFile.close();
        }
    }

    // 设置最小日志级别
    void setMinLogLevel(LogLevel level)
    {
        minLogLevel = level;
    }

    // 打印日志
    template <typename... Args>
    void log(LogLevel level, const char *file, int line, const char *format, Args... args)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (level >= minLogLevel)
        {
            std::stringstream ss;
            ss << "[" << getCurrentTime() << "] [" << logLevelToString(level) << "] [" << file << ":" << line << "] ";
            (ss << ... << args);

            std::string logMessage = ss.str();
            std::cout << logMessage << std::endl;
            if (logFile.is_open())
            {
                logFile << logMessage << std::endl;
            }
        }
    }

private:
    std::ofstream logFile;
    LogLevel minLogLevel;
    std::mutex mutex;
};

// 全局 logger 对象
static Logger g_logger("app.log", LogLevel::DEBUG);

// 辅助宏，用于处理可变参数列表
#define LOG_HELPER(level, ...) g_logger.log(level, __FILE__, __LINE__, "%s", ##__VA_ARGS__)

// 使用宏定义简化日志调用
#define LOG_DEBUG(...) LOG_HELPER(LogLevel::DEBUG, ##__VA_ARGS__)
#define LOG_INFO(...) LOG_HELPER(LogLevel::INFO, ##__VA_ARGS__)
#define LOG_WARNING(...) LOG_HELPER(LogLevel::WARNING, ##__VA_ARGS__)
#define LOG_ERROR(...) LOG_HELPER(LogLevel::ERROR, ##__VA_ARGS__)

#endif

