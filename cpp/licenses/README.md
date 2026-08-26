# HLIECpp third-party license bundle

This directory contains license and attribution material that must accompany
commercial binary distributions of HLIECpp.

The files are kept in the source repository instead of being collected only
from `third_party/` during packaging. Several CI-downloaded SDK archives are
replaced on every build, so relying on their directory contents would make the
published notices non-deterministic.

The current bundle covers the open-source runtime components shipped by the
package scripts, statically linked into HLIECpp, or compiled into HLIECpp from
headers. See `THIRD_PARTY_NOTICES.md` for the component-to-license mapping.

The package scripts validate `required-runtime-files.txt` and copy this
directory without modifying the license texts. Adding or replacing a runtime
dependency requires updating the notice, license text, and required-file list.
