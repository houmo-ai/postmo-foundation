#ifndef HLIE_GZIP_H
#define HLIE_GZIP_H

#include <filesystem>
#include <functional>

namespace hm_HLIE::detail {

// Compress source to source + ".gz" using an atomic temporary file. The source
// is removed only after the gzip file has been committed successfully.
bool gzip_log_file(const std::filesystem::path& source, int compression_level,
                   const std::function<bool()>& cancelled = {}) noexcept;

} // namespace hm_HLIE::detail

#endif // HLIE_GZIP_H
