"""main.py 的流程测试：用假的 AstrBot 接口，不需要真的启动 AstrBot。"""

import asyncio
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PLUGIN_DIR))
DATA_DIR = Path(tempfile.mkdtemp(prefix="reply_gate_test_"))


# ---------------------------------------------------------------- 假的 AstrBot
def _passthrough(*_args, **_kwargs):
    def wrap(func):
        return func

    return wrap


class _Logger:
    def __getattr__(self, _name):
        return lambda *a, **k: None


@dataclass
class DecisionBooleanQuestion:
    instructions: str


@dataclass
class DecisionBooleanAnswer:
    probability: float


class DecisionError(RuntimeError):
    pass


class Star:
    def __init__(self, context):
        self.context = context


class StarTools:
    @staticmethod
    def get_data_dir(_name):
        return DATA_DIR


class FakeRequest:
    body = {}
    query = {}

    async def json(self, default=None):
        return self.body if self.body is not None else default


REQUEST = FakeRequest()


def json_response(data=None, **_kw):
    return {"ok": True, **(data or {})}


def error_response(message, **_kw):
    return {"status": "error", "message": message}


class Image:
    """模仿 AstrBot 的 Image 组件。broken=True 时读取失败。"""

    def __init__(self, path="/tmp/fake.jpg", broken=False):
        self.path = path
        self.broken = broken

    async def convert_to_file_path(self):
        if self.broken:
            raise OSError("下载失败")
        return self.path


class Reply:
    """模仿 AstrBot 的 Reply 组件（引用回复）。"""

    def __init__(self, message_str="", sender_id="", sender_nickname=""):
        self.message_str = message_str
        self.sender_id = sender_id
        self.sender_nickname = sender_nickname


def _install_stubs():
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = _Logger()
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = types.SimpleNamespace(
        event_message_type=_passthrough,
        on_llm_response=_passthrough,
        after_message_sent=_passthrough,
        on_astrbot_loaded=_passthrough,
        permission_type=_passthrough,
        command=_passthrough,
        EventMessageType=types.SimpleNamespace(GROUP_MESSAGE=1),
        PermissionType=types.SimpleNamespace(ADMIN=1),
    )
    comps = types.ModuleType("astrbot.api.message_components")
    comps.Image = Image
    comps.Reply = Reply
    provider = types.ModuleType("astrbot.api.provider")
    provider.DecisionBooleanAnswer = DecisionBooleanAnswer
    provider.DecisionBooleanQuestion = DecisionBooleanQuestion
    provider.DecisionError = DecisionError
    provider.LLMResponse = object
    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = Star
    star.StarTools = StarTools
    web = types.ModuleType("astrbot.api.web")
    web.request = REQUEST
    web.json_response = json_response
    web.error_response = error_response
    sys.modules.update(
        {
            "astrbot": types.ModuleType("astrbot"),
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.message_components": comps,
            "astrbot.api.provider": provider,
            "astrbot.api.star": star,
            "astrbot.api.web": web,
        }
    )


_install_stubs()
from astrbot_plugin_reply_gate.main import ReplyGatePlugin  # noqa: E402

OPENID = "0123456789ABCDEF0123456789ABCDEF"


class FakeConfig(dict):
    saves = 0

    def save_config(self):
        FakeConfig.saves += 1


BOT_ID = "B" * 32
FRIEND_ID = "F" * 32


class At:
    """模仿 AstrBot 的 At 组件（类名必须叫 At）。"""

    def __init__(self, qq, name=""):
        self.qq = qq
        self.name = name


class FakeEvent:
    """raw: QQ 原始文字（含 <@openid>）。mentions: QQ 原始 mentions 列表。

    和 QQ 官方适配器一样：@ 机器人的标记会从 message_str 里去掉，并加一个 At 组件。
    """

    def __init__(self, text, sender_id="USER1", name="甲", wake=False, group="GROUP1",
                 raw=None, mentions=None):
        self.unified_msg_origin = f"bot:GroupMessage:{group}"
        self.session_id = group
        self.is_at_or_wake_command = wake
        self.call_llm = False
        self._sender_id = sender_id
        self._name = name
        self._group = group
        self._self_id = "BOT"
        self._comps = []
        self._extras = {}
        self._result = None
        if raw is not None:
            bot_marks = [m for m in (mentions or []) if m.get("is_you")]
            if bot_marks:
                self._self_id = bot_marks[0]["member_openid"]
                self._comps.append(At(self._self_id))
            text = raw
            for m in bot_marks:
                text = text.replace(f"<@{m['member_openid']}>", "")
            text = text.strip()
        self.message_str = text
        self.message_obj = types.SimpleNamespace(
            raw_message=types.SimpleNamespace(
                content=raw if raw is not None else text,
                raw_data={"mentions": mentions or []},
            )
        )

    def should_call_llm(self, value):
        self.call_llm = value

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return self._name

    def get_self_id(self):
        return self._self_id

    def get_group_id(self):
        return self._group

    def get_platform_id(self):
        return "bot"

    def get_platform_name(self):
        return "qq_official"

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_result(self):
        return self._result

    def get_messages(self):
        return list(self._comps)

    def request_llm(self, **kwargs):
        return ("LLM_REQUEST", kwargs)


class FakePersonaManager:
    def __init__(self, persona="小星"):
        self.persona = persona
        self.personas_v3 = [{"name": "default"}, {"name": "小星"}, {"name": "别人"}]
        self.calls = 0

    async def resolve_selected_persona(self, *, umo, conversation_persona_id, platform_name):
        self.calls += 1
        name = conversation_persona_id or self.persona
        found = next((p for p in self.personas_v3 if p["name"] == name), None)
        return name, found, None, False


class FakeContext:
    """decision: 概率 / 每个问题的概率 dict / 异常 / 'hang'（一直不返回）。

    只给一个数时，它是「被 @ 时要不要回」和「是不是对机器人说」的概率，另外两个问题是 0。
    llm: 对话模型回答的文字。
    """

    def __init__(
        self,
        decision=0.9,
        llm='{"reply": 0.9, "to_bot": 0.9, "to_other": 0.1, "open": 0.1, "new_topic": 0.1, "reason": "测试"}',
        persona="小星",
    ):
        self.decision_behaviour = decision
        self.llm_text = llm
        self.decision_calls = []
        self.decision_questions = []
        self.llm_calls = 0
        self.llm_kwargs = []
        self.conv_persona = None
        self.decision = types.SimpleNamespace(ask=self._ask)
        self.persona_manager = FakePersonaManager(persona)
        self.conversation_manager = types.SimpleNamespace(
            get_curr_conversation_id=self._cid,
            new_conversation=self._cid,
            get_conversation=self._conv,
        )
        self.provider_manager = types.SimpleNamespace(
            curr_decision_provider_inst=types.SimpleNamespace(
                provider_config={"id": "decision_1", "model": "jev-latest"}
            ),
            decision_provider_insts=[
                types.SimpleNamespace(
                    provider_config={"id": "decision_1", "model": "jev-latest", "provider": "custom"}
                )
            ],
            provider_insts=[types.SimpleNamespace(provider_config={"id": "flash", "model": "m"})],
        )
        self.astrbot_config_mgr = types.SimpleNamespace(
            confs={"default": {"provider_ltm_settings": {"active_reply": {"enable": False}}}}
        )
        self.web_routes = {}

    def register_web_api(self, route, handler, methods, desc):
        self.web_routes[(route, tuple(methods))] = handler

    async def _ask(self, questions, state, provider_id=None):
        self.decision_calls.append(state)
        self.decision_questions.append({k: q.instructions for k, q in questions.items()})
        behaviour = self.decision_behaviour
        if behaviour == "hang":
            await asyncio.sleep(60)
        if isinstance(behaviour, Exception):
            raise behaviour
        if isinstance(behaviour, dict):
            values = behaviour
        else:
            values = {"reply": behaviour, "to_bot": behaviour}
        return {k: DecisionBooleanAnswer(float(values.get(k, 0.0))) for k in questions}

    async def llm_generate(self, **kwargs):
        self.llm_calls += 1
        self.llm_kwargs.append(kwargs)
        return types.SimpleNamespace(completion_text=self.llm_text)

    async def get_using_provider_async(self, _umo):
        return types.SimpleNamespace(meta=lambda: types.SimpleNamespace(id="chat"))

    async def _cid(self, *_args):
        return "cid"

    async def _conv(self, *_args):
        return types.SimpleNamespace(persona_id=self.conv_persona)


def _plugin(context, **config):
    for f in DATA_DIR.iterdir():
        f.unlink()
    base = FakeConfig(
        {"persona_aliases": {"小星": "星星"}, "cooldown_seconds": 30, "judge_timeout": 0.2}
    )
    base.update(config)
    return ReplyGatePlugin(context, base)


async def _send(plugin, event):
    return [item async for item in plugin.on_group_message(event)]


async def _call(plugin, path, body=None):
    REQUEST.body = body or {}
    method = ("POST",) if body is not None else ("GET",)
    handler = plugin.context.web_routes[(f"/astrbot_plugin_reply_gate/{path}", method)]
    return await handler()


# ---------------------------------------------------------------- 判断流程
async def test_blacklist_never_asks_model():
    ctx = FakeContext()
    plugin = _plugin(ctx, user_blacklist=[OPENID])
    out = await _send(plugin, FakeEvent("小明，你作业写完了吗", OPENID, "小红"))
    assert out == [] and ctx.decision_calls == [] and ctx.llm_calls == 0
    assert plugin.history[-1]["stage"] == "用户黑名单"


async def test_wake_message_is_ignored():
    ctx = FakeContext()
    plugin = _plugin(ctx)
    assert await _send(plugin, FakeEvent("你好", wake=True)) == []
    assert ctx.decision_calls == [] and len(plugin.history) == 0
    assert "USER1" in plugin.roster.users  # 但发言的人会被记下来


async def test_high_probability_replies_then_cooldown():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx)
    out = await _send(plugin, FakeEvent("今天有空吗"))
    assert len(out) == 1 and out[0][0] == "LLM_REQUEST"
    assert out[0][1]["prompt"] == "今天有空吗"
    assert plugin.history[-1]["probability"] == 0.9
    assert plugin.history[-1]["stage"] == "对机器人说"
    # 冷却期间：对机器人说的话（追问）照样回复
    out = await _send(plugin, FakeEvent("那明天呢"))
    assert len(out) == 1 and len(ctx.decision_calls) == 2
    # 叫了别称：名字会标出来
    out = await _send(plugin, FakeEvent("星星，那后天呢"))
    assert len(out) == 1
    assert ctx.decision_calls[-1]["current_message"]["mentions_bot_name"] is True


async def test_open_question_uses_chance_and_cooldown():
    ctx = FakeContext(decision={"to_bot": 0.1, "to_other": 0.1, "open": 0.9})
    plugin = _plugin(ctx, open_question_chance=1.0)
    assert len(await _send(plugin, FakeEvent("吃了吗"))) == 1
    assert plugin.history[-1]["stage"] == "开放问题"
    # 冷却期间：开放问题不接，但仍然问了模型（万一是对机器人说的）
    assert await _send(plugin, FakeEvent("有人知道附近哪能修电脑吗")) == []
    assert plugin.history[-1]["stage"] == "冷却中" and len(ctx.decision_calls) == 2
    # 接话概率为 0：从不接
    quiet = _plugin(FakeContext(decision={"open": 0.9}), open_question_chance=0.0)
    assert await _send(quiet, FakeEvent("吃了吗")) == []
    assert quiet.history[-1]["stage"] == "开放问题"
    # 看起来像开放问题，但更像是对别人说的：不接
    other = _plugin(FakeContext(decision={"open": 0.8, "to_other": 0.8}), open_question_chance=1.0)
    assert await _send(other, FakeEvent("吃了吗")) == []
    assert other.history[-1]["stage"] == "对别人说"


async def test_new_topic_can_be_joined():
    ctx = FakeContext(decision={"to_bot": 0.05, "to_other": 0.1, "open": 0.5, "new_topic": 0.9})
    plugin = _plugin(ctx, open_question_chance=1.0)
    assert len(await _send(plugin, FakeEvent("今天终于发工资了！"))) == 1
    assert plugin.history[-1]["stage"] == "新话题"
    assert set(ctx.decision_questions[-1]) == {"to_bot", "to_other", "open", "new_topic"}


async def test_open_threshold_is_configurable():
    scores = {"to_bot": 0.05, "to_other": 0.1, "open": 0.65, "new_topic": 0.2}
    loose = _plugin(FakeContext(decision=scores), open_question_chance=1.0)
    assert len(await _send(loose, FakeEvent("周末大家去哪玩"))) == 1
    assert loose.history[-1]["stage"] == "开放问题"
    strict = _plugin(FakeContext(decision=scores), open_question_chance=1.0, open_threshold=0.8)
    assert await _send(strict, FakeEvent("周末大家去哪玩")) == []
    assert strict.history[-1]["stage"] == "无需回应"


async def test_to_other_beats_to_bot():
    ctx = FakeContext(decision={"to_bot": 0.7, "to_other": 0.9})
    plugin = _plugin(ctx)
    assert await _send(plugin, FakeEvent("喝过，一般般")) == []
    assert plugin.history[-1]["stage"] == "对别人说"


async def test_quote_of_bot_message_reaches_model():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await plugin.on_llm_response(
        FakeEvent("x", name="乙"), types.SimpleNamespace(completion_text="碧螺春吧，清淡些。")
    )
    await _send(plugin, FakeEvent("丙说点别的", name="丙"))
    event = FakeEvent("为什么")
    event._comps.append(Reply("碧螺春吧，清淡些。"))
    await _send(plugin, event)
    state = ctx.decision_calls[-1]
    assert state["current_message"]["quoting"] == {"speaker": "机器人自己", "text": "碧螺春吧，清淡些。"}
    assert state["recent_messages"][0]["replying_to"] == "乙"
    # 引用的是别人的话：标出是谁
    event = FakeEvent("真的吗")
    event._comps.append(Reply("丙说点别的"))
    await _send(plugin, event)
    assert ctx.decision_calls[-1]["current_message"]["quoting"]["speaker"] == "丙"


async def test_names_follow_persona():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("今天好冷"))
    state = ctx.decision_calls[-1]
    assert state["bot_name"] == "小星" and state["bot_aliases"] == ["星星"]
    assert plugin.roster.groups["GROUP1"]["persona"] == "小星"
    # 对话切换到另一个人格后（缓存过期），名字跟着变，旧别称不再算
    ctx.conv_persona = "别人"
    plugin._persona_cache.clear()
    await _send(plugin, FakeEvent("星星在吗"))
    state = ctx.decision_calls[-1]
    assert state["bot_name"] == "别人" and state["bot_aliases"] == []
    assert state["current_message"]["mentions_bot_name"] is False


async def test_persona_lookup_is_cached():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    for text in ("第一句话", "第二句话", "第三句话"):
        await _send(plugin, FakeEvent(text))
    assert ctx.persona_manager.calls == 1


async def test_no_persona_uses_extra_names():
    ctx = FakeContext(decision=0.1, persona="default")
    plugin = _plugin(ctx, extra_names=["小助手"])
    await _send(plugin, FakeEvent("小助手在吗"))
    state = ctx.decision_calls[-1]
    assert state["bot_name"] == "小助手" and state["current_message"]["mentions_bot_name"]


async def test_low_probability_stays_quiet():
    ctx = FakeContext(decision=0.2)
    plugin = _plugin(ctx)
    assert await _send(plugin, FakeEvent("你们晚上去哪吃饭")) == []
    assert plugin.history[-1]["stage"] == "无需回应" and ctx.llm_calls == 0


async def test_threshold_is_configurable():
    ctx = FakeContext(decision=0.5)
    assert await _send(_plugin(ctx, threshold=0.6), FakeEvent("这个怎么卖")) == []
    assert len(await _send(_plugin(ctx, threshold=0.4), FakeEvent("这个怎么卖"))) == 1


async def test_decision_error_falls_back_to_llm():
    ctx = FakeContext(decision=DecisionError("not configured"))
    plugin = _plugin(ctx)
    assert len(await _send(plugin, FakeEvent("有人在吗"))) == 1
    item = plugin.history[-1]
    assert ctx.llm_calls == 1 and item["stage"] == "对机器人说"
    assert "对话模型" in item["detail"] and "判断模型不可用" in item["detail"]


async def test_timeout_and_garbage_means_no_reply():
    ctx = FakeContext(decision="hang", llm="我不知道")
    plugin = _plugin(ctx)
    assert await _send(plugin, FakeEvent("有人在吗")) == []
    assert plugin.history[-1]["stage"] == "判断失败"
    plugin = _plugin(ctx, reply_on_error=True)
    assert len(await _send(plugin, FakeEvent("有人在吗"))) == 1


async def test_fallback_can_be_disabled():
    ctx = FakeContext(decision=DecisionError("down"))
    plugin = _plugin(ctx, llm_fallback_enabled=False)
    assert await _send(plugin, FakeEvent("有人在吗")) == []
    assert ctx.llm_calls == 0


async def test_context_includes_bot_reply():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("星星你几点下班", wake=True))
    await plugin.on_llm_response(
        FakeEvent("x"), types.SimpleNamespace(completion_text="晚上十点。")
    )
    await _send(plugin, FakeEvent("这么晚啊"))
    recent = ctx.decision_calls[-1]["recent_messages"]
    assert [m["text"] for m in recent] == ["星星你几点下班", "晚上十点。"]
    assert recent[1]["speaker"] == "机器人自己"


class FakeResult:
    def __init__(self, text):
        self.text = text

    def get_plain_text(self):
        return self.text


async def test_sent_messages_recorded_once():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    # 对话模型的回复：两个钩子都会触发，只记一次
    event = FakeEvent("星星几点了", wake=True)
    await _send(plugin, event)
    await plugin.on_llm_response(event, types.SimpleNamespace(completion_text="三点。"))
    event._result = FakeResult("三点。")
    await plugin.after_message_sent(event)
    # 不经过对话模型的回复（比如指令结果）也要记
    other = FakeEvent("/帮助", name="乙")
    other._result = FakeResult("可用指令：……")
    await plugin.after_message_sent(other)
    await _send(plugin, FakeEvent("好的知道了谢谢"))
    recent = ctx.decision_calls[-1]["recent_messages"]
    assert [m["text"] for m in recent] == ["星星几点了", "三点。", "可用指令：……"]
    assert recent[1]["replying_to"] == "甲" and recent[2]["replying_to"] == "乙"


async def test_context_survives_restart():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("第一句话", name="乙"))
    await plugin.on_llm_response(FakeEvent("x", name="乙"), types.SimpleNamespace(completion_text="收到。"))
    await plugin.terminate()
    again = ReplyGatePlugin(ctx, FakeConfig({"persona_aliases": {"小星": "星星"}}))
    await _send(again, FakeEvent("第二句话"))
    recent = ctx.decision_calls[-1]["recent_messages"]
    assert [m["text"] for m in recent] == ["第一句话", "收到。"]
    assert recent[1]["speaker"] == "机器人自己" and recent[1]["replying_to"] == "乙"
    assert again.history[-1]["context_count"] == 2


def _image_event(text="", *paths, broken=False):
    event = FakeEvent(text)
    for path in paths or ("/tmp/a.jpg",):
        event._comps.append(Image(path, broken=broken))
    return event


async def test_image_message_goes_to_llm_with_picture():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx, debug_enabled=True)
    out = await _send(plugin, _image_event("", "/tmp/a.jpg", "/tmp/b.jpg"))
    # 纯图片也不算「没有实质内容」；判断模型看不了图，所以只问了对话模型
    assert len(out) == 1 and ctx.decision_calls == [] and ctx.llm_calls == 1
    sent = ctx.llm_kwargs[-1]
    assert sent["image_urls"] == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert "附带了 2 张图片" in sent["prompt"] and "[图片]×2" in sent["prompt"]
    record = plugin.debug[-1]
    assert record["images"] == 2 and record["images_sent"] == 2
    assert record["judge_text"] == "[图片]×2" and "带 2 张图" in record["model"]
    # 后面的消息能在前文里看到这张图
    await _send(plugin, FakeEvent("好看吗"))
    assert ctx.llm_calls == 1 and len(ctx.decision_calls) == 1  # 没带图的消息照常走判断模型
    assert ctx.decision_calls[-1]["recent_messages"][-1]["text"] == "[图片]×2"


async def test_image_falls_back_to_decision_when_llm_fails():
    ctx = FakeContext(decision=0.9, llm="看不懂的回答")
    plugin = _plugin(ctx)
    assert len(await _send(plugin, _image_event("这是啥"))) == 1
    item = plugin.history[-1]
    assert ctx.llm_calls == 1 and len(ctx.decision_calls) == 1
    assert "判断模型看不到图片" in item["detail"]
    assert ctx.decision_calls[-1]["current_message"]["images"] == 1
    # 图片下载失败：同样退回判断模型
    ctx2 = FakeContext(decision=0.9)
    plugin2 = _plugin(ctx2)
    await _send(plugin2, _image_event("这是啥", broken=True))
    assert len(ctx2.decision_calls) == 1 and "图片都读取失败" in plugin2.history[-1]["detail"]


async def test_image_to_llm_can_be_turned_off():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx, image_use_llm=False)
    assert len(await _send(plugin, _image_event("这是啥"))) == 1
    assert len(ctx.decision_calls) == 1 and ctx.llm_calls == 0


async def test_disabled_plugin_does_nothing():
    ctx = FakeContext()
    plugin = _plugin(ctx, enabled=False)
    assert await _send(plugin, FakeEvent("有人在吗")) == []
    assert ctx.decision_calls == []


async def test_judge_off_uses_name_and_probability():
    ctx = FakeContext()
    plugin = _plugin(ctx, judge_enabled=False)
    assert await _send(plugin, FakeEvent("今天天气不错")) == []
    assert len(await _send(plugin, FakeEvent("星星今天天气不错"))) == 1
    assert ctx.decision_calls == [] and ctx.llm_calls == 0


# ---------------------------------------------------------------- 面板接口
async def test_api_routes_registered():
    plugin = _plugin(FakeContext())
    paths = {route for route, _ in plugin.context.web_routes}
    for name in ("overview", "config", "config/save", "people", "list/toggle", "history/clear", "test"):
        assert f"/astrbot_plugin_reply_gate/{name}" in paths


async def test_api_blacklist_from_people():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("随便聊聊天", OPENID, "小红"))
    people = await _call(plugin, "people")
    user = next(u for u in people["users"] if u["id"] == OPENID)
    assert user["name"] == "小红" and not user["blacklisted"]
    res = await _call(plugin, "list/toggle", {"list": "user", "id": OPENID})
    assert res["on"] is True and plugin.config["user_blacklist"] == [OPENID]
    assert (await _call(plugin, "people"))["users"][0]["blacklisted"]
    calls = len(ctx.decision_calls)
    await _send(plugin, FakeEvent("再说一句话", OPENID, "小红"))
    assert len(ctx.decision_calls) == calls  # 拉黑后立即生效，不再调用模型
    res = await _call(plugin, "list/toggle", {"list": "user", "id": OPENID.lower(), "on": False})
    assert res["items"] == []
    bad = await _call(plugin, "list/toggle", {"list": "nope", "id": "x"})
    assert bad["status"] == "error"


async def test_api_save_config_validates():
    plugin = _plugin(FakeContext())
    FakeConfig.saves = 0
    res = await _call(
        plugin,
        "config/save",
        {
            "config": {
                "threshold": "1.7",
                "cooldown_seconds": "45",
                "min_length": 3.9,
                "persona_aliases": {"小星": ["星星", "管家"], "空的": []},
                "extra_names": "甲，乙",
                "judge_enabled": False,
                "not_a_key": 1,
            }
        },
    )
    cfg = plugin.config
    assert cfg["threshold"] == 1.0 and cfg["cooldown_seconds"] == 45.0 and cfg["min_length"] == 3
    assert cfg["persona_aliases"] == {"小星": "星星，管家"}
    assert cfg["extra_names"] == ["甲", "乙"] and cfg["judge_enabled"] is False
    assert "not_a_key" not in cfg and FakeConfig.saves == 1
    assert res["config"]["persona_aliases"] == {"小星": ["星星", "管家"]}
    again = await _call(plugin, "config/save", {"config": {"threshold": 1.0}})
    assert again["changed"] == [] and FakeConfig.saves == 1
    bad = await _call(plugin, "config/save", {"config": {"threshold": "abc"}})
    assert bad["status"] == "error"


async def test_api_config_and_overview():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("有人在吗"))
    cfg = await _call(plugin, "config")
    assert cfg["personas"] == ["小星", "别人"]  # 系统默认人格不列出
    assert cfg["providers"]["decision"][0]["id"] == "decision_1"
    overview = await _call(plugin, "overview")
    assert overview["stats"]["total"] == 1 and overview["stats"]["replied"] == 1
    assert overview["history"][0]["speaker"] == "甲" and overview["warnings"] == []
    ctx.astrbot_config_mgr.confs["abc12345xyz"] = {
        "provider_ltm_settings": {"active_reply": {"enable": True}}
    }
    assert "内置" in (await _call(plugin, "overview"))["warnings"][0]
    await _call(plugin, "history/clear", {})
    assert (await _call(plugin, "overview"))["history"] == []


async def test_api_test_endpoint():
    ctx = FakeContext(decision=0.92)
    plugin = _plugin(ctx, user_blacklist=["USER1"])
    res = await _call(
        plugin,
        "test",
        {
            "text": "星星那明天呢",
            "speaker": "甲",
            "persona": "小星",
            "context": [{"speaker": "机器人", "text": "晚上十点。", "is_bot": True}],
        },
    )
    assert res["reply"] and res["probability"] == 0.92 and res["named"]
    assert res["names"] == ["小星", "星星"]
    assert ctx.decision_calls[-1]["recent_messages"][0]["speaker"] == "机器人自己"
    noise = await _call(plugin, "test", {"text": "哈哈哈哈"})
    assert noise["stage"] == "没有实质内容" and not noise["reply"]
    empty = await _call(plugin, "test", {"text": " "})
    assert empty["status"] == "error"


async def test_state_survives_restart():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("第一次见面", OPENID, "小红"))
    await plugin.terminate()
    saved = json.loads((DATA_DIR / "state.json").read_text(encoding="utf-8"))
    assert OPENID in saved["roster"]["users"] and len(saved["history"]) == 1
    again = ReplyGatePlugin(ctx, FakeConfig())
    assert again.roster.name_of(OPENID) == "小红" and len(again.history) == 1


# ---------------------------------------------------------------- @ 的处理
def _qq_mentions(*, bot=False, friend=True, friend_name=""):
    items = []
    if friend:
        item = {"member_openid": FRIEND_ID, "is_you": False}
        if friend_name:
            item["username"] = friend_name
        items.append(item)
    if bot:
        items.append({"member_openid": BOT_ID, "is_you": True})
    return items


async def test_mention_becomes_nickname_for_idle_message():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    # 小明先在群里说过话，插件记得他的昵称
    await _send(plugin, FakeEvent("大家好", FRIEND_ID, "小明"))
    event = FakeEvent("", raw=f"<@{FRIEND_ID}>，和星星说句话", mentions=_qq_mentions())
    assert await _send(plugin, event) == []
    current = ctx.decision_calls[-1]["current_message"]
    assert current["text"] == "@小明，和星星说句话"
    assert current["mentions"] == ["小明"] and "at_bot" not in current
    # 上下文里存的也是换好的文字
    later = FakeEvent("那你们聊")
    await _send(plugin, later)
    texts = [m["text"] for m in ctx.decision_calls[-1]["recent_messages"]]
    assert "@小明，和星星说句话" in texts


async def test_mention_name_from_raw_payload_and_unknown():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("", raw=f"<@{FRIEND_ID}> 晚上见",
                                  mentions=_qq_mentions(friend_name="阿强")))
    assert ctx.decision_calls[-1]["current_message"]["text"] == "@阿强 晚上见"
    await _send(plugin, FakeEvent("", raw="<@ZZZZ> 明天见", mentions=[]))
    assert ctx.decision_calls[-1]["current_message"]["mentions"] == ["某位群友"]


async def test_at_bot_and_others_low_score_blocks_default_reply():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    event = FakeEvent("", wake=True, raw=f"<@{FRIEND_ID}>，和<@{BOT_ID}> 说句话",
                      mentions=_qq_mentions(bot=True, friend_name="小明"))
    assert event.message_str == f"<@{FRIEND_ID}>，和 说句话"  # 适配器去掉了机器人的标记
    assert await _send(plugin, event) == []
    state = ctx.decision_calls[-1]
    assert state["current_message"]["text"] == "@小明，和@小星 说句话"
    assert state["current_message"]["at_bot"] and state["current_message"]["mentions"] == ["小明"]
    assert event.call_llm is True  # 默认回复被拦下
    assert plugin.history[-1]["path"] == "被@时判断" and not plugin.history[-1]["reply"]


async def test_at_bot_and_others_high_score_keeps_reply():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx)
    event = FakeEvent("", wake=True, raw=f"<@{BOT_ID}> 你帮我劝劝<@{FRIEND_ID}>",
                      mentions=_qq_mentions(bot=True, friend_name="小明"))
    out = await _send(plugin, event)
    assert out == [] and event.call_llm is False  # 不自己发请求，交给 AstrBot 照常回复
    assert plugin.history[-1]["reply"]


async def test_at_bot_judge_failure_keeps_reply():
    ctx = FakeContext(decision="hang", llm="看不懂")
    plugin = _plugin(ctx)
    event = FakeEvent("", wake=True, raw=f"<@{FRIEND_ID}> 和<@{BOT_ID}> 说话",
                      mentions=_qq_mentions(bot=True))
    await _send(plugin, event)
    assert event.call_llm is False and plugin.history[-1]["stage"] == "判断失败"


async def test_at_bot_only_or_switch_off_is_not_judged():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    only_bot = FakeEvent("", wake=True, raw=f"<@{BOT_ID}> 你好",
                         mentions=_qq_mentions(bot=True, friend=False))
    await _send(plugin, only_bot)
    assert ctx.decision_calls == [] and only_bot.call_llm is False
    command = FakeEvent("/help", wake=True)
    await _send(plugin, command)
    assert ctx.decision_calls == []
    plugin = _plugin(ctx, judge_when_at_with_others=False)
    both = FakeEvent("", wake=True, raw=f"<@{FRIEND_ID}>，和<@{BOT_ID}> 说句话",
                     mentions=_qq_mentions(bot=True))
    await _send(plugin, both)
    assert ctx.decision_calls == [] and both.call_llm is False


async def test_blacklisted_user_still_gets_reply_when_at_bot():
    ctx = FakeContext(decision=0.9)
    plugin = _plugin(ctx, user_blacklist=[OPENID])
    event = FakeEvent("", sender_id=OPENID, wake=True,
                      raw=f"<@{BOT_ID}> 帮我问问<@{FRIEND_ID}>",
                      mentions=_qq_mentions(bot=True))
    await _send(plugin, event)
    assert event.call_llm is False and plugin.history[-1]["stage"] == "判断模型"


# ---------------------------------------------------------------- 调试记录
async def test_debug_off_records_nothing():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    await _send(plugin, FakeEvent("今天好冷"))
    assert len(plugin.debug) == 0 and plugin.history[-1]["has_debug"] is False
    assert (await _call(plugin, "debug"))["enabled"] is False


async def test_debug_on_keeps_full_trace():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx, debug_enabled=True)
    await _send(plugin, FakeEvent("前一句话"))
    await _send(plugin, FakeEvent("", raw=f"<@{FRIEND_ID}>，和星星说句话",
                                  mentions=_qq_mentions(friend_name="小明")))
    await _send(plugin, FakeEvent("哈哈哈哈"))  # 规则过滤掉的也要记
    records = (await _call(plugin, "debug"))["records"]
    assert len(records) == 3
    noise, mention = records[0], records[1]  # 最新的在前
    assert noise["stage"] == "没有实质内容" and noise["model_input"] is None
    assert mention["raw_text"] == f"<@{FRIEND_ID}>，和星星说句话"
    assert mention["judge_text"] == "@小明，和星星说句话" and mention["mentions"] == ["小明"]
    assert mention["context"][0]["text"] == "前一句话"
    assert "【对机器人说】" in mention["question"] and "【对全群说】" in mention["question"]
    assert mention["scores"] == {"to_bot": 0.1, "to_other": 0.0, "open": 0.0, "new_topic": 0.0}
    assert "【新话题】" in mention["question"]
    assert mention["probability"] == 0.1 and mention["context"][0]["time"].endswith("秒前")
    assert mention["model"].startswith("判断模型") and mention["elapsed_ms"] is not None
    assert plugin.history[-2]["has_debug"] and plugin.history[-2]["id"] == mention["id"]
    one = await _call(plugin, "debug", None)
    assert one["count"] == 3


async def test_debug_records_persist_resize_and_clear():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx, debug_enabled=True)
    for i in range(5):
        await _send(plugin, FakeEvent(f"第{i}句话来了"))
    await plugin.terminate()
    saved = json.loads((DATA_DIR / "debug.json").read_text(encoding="utf-8"))
    assert len(saved) == 5
    again = ReplyGatePlugin(ctx, FakeConfig({"debug_enabled": True}))
    assert len(again.debug) == 5
    await _call(again, "config/save", {"config": {"debug_max_records": 3}})
    assert again.debug.maxlen == 10  # 最小 10 条
    await _call(again, "config/save", {"config": {"debug_max_records": 20}})
    assert again.debug.maxlen == 20 and len(again.debug) == 5
    exported = await _call(again, "debug/export")
    assert len(exported["records"]) == 5
    await _call(again, "debug/clear", {})
    assert len(again.debug) == 0 and not any(h["has_debug"] for h in again.history)
    assert json.loads((DATA_DIR / "debug.json").read_text(encoding="utf-8")) == []


async def test_invisible_nickname_falls_back_to_id():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx, debug_enabled=True)
    await _send(plugin, FakeEvent("今天天气不错", "C0FFEE12XXXX", "‭"))
    assert plugin.history[-1]["speaker"] == "用户C0FFEE"
    assert plugin.debug[-1]["speaker"] == "用户C0FFEE"
    assert ctx.decision_calls[-1]["current_message"]["speaker"] == "用户C0FFEE"


async def test_api_test_handles_written_mentions():
    ctx = FakeContext(decision=0.1)
    plugin = _plugin(ctx)
    res = await _call(plugin, "test", {"text": "@小明，和@小星 说句话", "persona": "小星"})
    assert res["path"] == "被@时判断" and res["mentions"] == ["小明"] and not res["reply"]
    assert res["model_input"]["current_message"]["at_bot"] is True
    res = await _call(plugin, "test", {"text": "@小明 晚上吃什么", "persona": "小星"})
    assert res["path"] == "主动接话" and res["mentions"] == ["小明"]


# ---------------------------------------------------------------- 重启后恢复群会话场景
class FakePlatform:
    """模仿 qq_official 适配器：场景只存在内存里，重启就没了。"""

    def __init__(self, pid="bot", ptype="qq_official"):
        self._meta = types.SimpleNamespace(id=pid, name=ptype)
        self._session_scene = {}

    def meta(self):
        return self._meta

    def remember_session_scene(self, session_id, scene):
        self._session_scene[session_id] = scene


def _with_platforms(ctx, *platforms):
    ctx.platform_manager = types.SimpleNamespace(platform_insts=list(platforms))
    return ctx


async def test_group_scene_restored_after_restart():
    platform = FakePlatform()
    ctx = _with_platforms(FakeContext(decision=0.1), platform)
    plugin = _plugin(ctx)
    platform._session_scene["CHAN1"] = "channel"  # 适配器在派发消息前已经记好场景
    await _send(plugin, FakeEvent("频道里说句话", OPENID, "小红", group="CHAN1"))
    await plugin.terminate()
    assert plugin.roster.groups["CHAN1"]["scene"] == "channel"

    fresh = FakePlatform()  # 重启：适配器缓存清空
    again = ReplyGatePlugin(_with_platforms(FakeContext(), fresh), FakeConfig())
    await again.on_astrbot_loaded()
    assert fresh._session_scene == {"CHAN1": "channel"}


async def test_legacy_groups_without_scene_are_inferred_conservatively():
    hex_id, numeric_id = "A" * 32, "1234567890"
    platform = FakePlatform()
    plugin = _plugin(_with_platforms(FakeContext(decision=0.1), platform))
    for gid in (hex_id, numeric_id, "OTHERPLATFORMGROUP"):
        plugin.roster.see("u", "甲", gid, f"bot:GroupMessage:{gid}")
    plugin.roster.groups["OTHERPLATFORMGROUP"]["umo"] = "other:GroupMessage:OTHERPLATFORMGROUP"
    platform._session_scene["KEEP"] = "channel"
    plugin.roster.see("u", "甲", "KEEP", "bot:GroupMessage:KEEP")
    plugin.roster.note_scene("KEEP", "group")
    assert plugin._restore_group_scenes() == 1
    # 32 位十六进制当 QQ 群；纯数字认不出就不猜；别的平台不碰；已有缓存不覆盖
    assert platform._session_scene == {"KEEP": "channel", hex_id: "group"}


async def test_restore_skips_platforms_without_scene_cache():
    other = types.SimpleNamespace(meta=lambda: types.SimpleNamespace(id="bot", name="aiocqhttp"))
    plugin = _plugin(_with_platforms(FakeContext(), other))
    plugin.roster.see("u", "甲", "A" * 32, "bot:GroupMessage:" + "A" * 32)
    assert plugin._restore_group_scenes() == 0
    no_manager = _plugin(FakeContext())
    await no_manager.initialize()  # 正常启动时平台还没加载，不能出错


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        asyncio.run(test())
        print("ok", test.__name__)
    print(f"{len(tests)} 个流程测试全部通过")
