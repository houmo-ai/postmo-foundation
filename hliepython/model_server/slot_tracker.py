"""
并发槽位跟踪器：让 hliepython worker 进程能够：
1. 跟踪 in-flight 请求数
2. 给 hibalance Agent 提供 slots_idle / slots_processing 数值
3. 给 hibalance Agent 提供 /slots 数组（hi.cpp 兼容格式）
"""
import asyncio
import time
from typing import Optional


class SlotTracker:
    """
    单个 worker 进程内的并发槽位跟踪器。

    - max_slots 取自 vllm engine.max-num-seqs（或配置 max-num-seqs）。
    - 当前空闲槽位 = semaphore._value。
    - 提供 hibalance 兼容的 slots_idle / slots_processing / slots 数组。
    """

    def __init__(self, max_slots: int = 1):
        if max_slots < 1:
            max_slots = 1
        self.max_slots = max_slots
        self.semaphore = asyncio.Semaphore(max_slots)
        self._active_count = 0
        # 记录每个 slot 启动时间（用于排错/可观测）
        self._slot_started_at: dict[int, float] = {}
        # 用于 acquire 时分配唯一 slot id
        self._lock = asyncio.Lock()

    @property
    def slots_idle(self) -> int:
        """当前空闲槽位数。hibalance 选路时使用。"""
        return max(self.semaphore._value, 0)

    @property
    def slots_processing(self) -> int:
        return self.max_slots - self.slots_idle

    @property
    def is_ready(self) -> bool:
        return self.semaphore is not None

    def aggregated_status(self) -> dict:
        """hibalance Agent 上报格式（POST /register/target 使用）"""
        return {
            "status": "ok" if self.slots_idle > 0 else "busy",
            "slots_idle": self.slots_idle,
            "slots_processing": self.slots_processing,
            "error_message": "",
        }

    def to_slots_array(self, n_ctx: int = 0) -> list[dict]:
        """hi.cpp 兼容的 /slots 端点返回格式。

        hibalance Agent 会累计 is_processing 字段得到 slots_idle/slots_processing。
        """
        idle = self.slots_idle
        slots = []
        for i in range(self.max_slots):
            slots.append({
                "id": i,
                "n_ctx": n_ctx,
                "is_processing": i >= idle,  # 前 idle 个是空闲
                "speculative": False,
                "task_id": -1,
                "state": 1 if i >= idle else 0,
            })
        return slots

    async def acquire(self) -> int:
        """获取一个空闲 slot，返回 slot id。"""
        await self.semaphore.acquire()
        async with self._lock:
            self._active_count += 1
            # 找一个空闲 slot id
            for i in range(self.max_slots):
                if i not in self._slot_started_at:
                    self._slot_started_at[i] = time.time()
                    return i
            return self.max_slots - 1

    def release(self, slot_id: Optional[int] = None) -> None:
        try:
            self.semaphore.release()
        except ValueError:
            # 重复 release 容错
            pass
        self._active_count = max(0, self._active_count - 1)
        if slot_id is not None:
            self._slot_started_at.pop(slot_id, None)

    def shutdown(self) -> None:
        """worker 退出时清理。"""
        self._slot_started_at.clear()
        self._active_count = 0
