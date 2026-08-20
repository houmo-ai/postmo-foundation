#ifndef HLIE_GZIP_H
#define HLIE_GZIP_H

#include <filesystem>
#include <functional>
#include <string>

#include "hlie_base/hlie_base.h"

namespace hm_HLIE::detail {

// Compress source to source + ".gz" using an atomic temporary file. The source
// is removed only after the gzip file has been committed successfully.
HLIE_API bool gzip_log_file(const std::filesystem::path& source, int compression_level,
                            const std::function<bool()>& cancelled = {}) noexcept;

// Read one gzip file through the same zlib instance owned by hlie-base. Keeping
// this behind hlie-base prevents consumers and tests from statically linking a
// second zlib copy into the same process.
HLIE_API bool read_gzip_file(const std::filesystem::path& source,
                             std::string& output) noexcept;

} // namespace hm_HLIE::detail

#endif // HLIE_GZIP_H
