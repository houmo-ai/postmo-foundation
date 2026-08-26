#include <catch_amalgamated.hpp>

#include <memory>
#include <string>
#include <utility>

#include "hlie_app/hlie_model_service.h"
#include "hlie_app/runtime/hlie_backend_adapter_registry.h"

using namespace hm_HLIE;

namespace {

class FakeModelService final : public IHLIEModelService {
public:
    int init(const std::string&, const HLIEModelInfo&,
             const std::shared_ptr<HLIEDeviceService>&) override {
        return 0;
    }
    void run() override {}
    void process() override {}
    void stop() override {}
    int get_state() override { return 0; }
};

class FakeBackendAdapter final : public IHLIEBackendAdapter {
public:
    explicit FakeBackendAdapter(std::string id) : id_(std::move(id)) {}

    [[nodiscard]] std::string backend_id() const override { return id_; }
    [[nodiscard]] std::shared_ptr<IHLIEModelService>
    create_embedded_service() const override {
        return std::make_shared<FakeModelService>();
    }

private:
    std::string id_;
};

} // namespace

TEST_CASE("backend adapter registry joins stable ids to app service factories",
          "[p1][app][backend-adapter]") {
    HLIEBackendAdapterRegistry registry;
    REQUIRE(registry.register_adapter(
        std::make_shared<FakeBackendAdapter>("houmo.fake")));
    CHECK(registry.size() == 1);
    CHECK(registry.find("  HOUMO.FAKE  ") != nullptr);
    CHECK(registry.create_embedded_service("houmo.fake") != nullptr);
    CHECK(registry.create_embedded_service("houmo.missing") == nullptr);

    CHECK_FALSE(registry.register_adapter(
        std::make_shared<FakeBackendAdapter>("HOUMO.FAKE")));
    CHECK_FALSE(registry.register_adapter(nullptr));
    CHECK_FALSE(registry.register_adapter(
        std::make_shared<FakeBackendAdapter>("   ")));
}
