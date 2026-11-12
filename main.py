"""回复判断：由判断模型决定机器人要不要在群里接话。

- 没有 @ 机器人的群消息：先按名单和规则过滤，再交给模型判断要不要主动接话。
- @ 了机器人、同时也 @ 了别人的消息：交给模型判断这句话是不是真的对机器人说的，
  不是的话拦掉 AstrBot 的默认回复（可在设置里关闭）。
- 其它被 @ 或被唤醒的消息：照常回复，不经过判断。

机器人的名字取自当前会话生效的人格，别称在面板里按人格配置。
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.provider import (
    DecisionBooleanAnswer,
    DecisionBooleanQuestion,
    DecisionError,
    LLMResponse,
)
from astrbot.api.star import Context, Star

from .gate_logic import (
    _MENTION_RE,
    BOT_SPEAKER,
    LLM_SYSTEM_PROMPT,
    UNKNOWN_MEMBER,
    ChatLine,
    ContextBuffer,
    Roster,
    Scores,
    Verdict,
    build_at_question,
    build_llm_prompt,
    build_questions,
    build_state,
    clean_name,
    combine,
    compose_names,
    is_noise,
    mentions_name,
    normalize_mentions,
    parse_llm_scores,
    prefilter,
    questions_text,
    split_names,
    with_image_mark,
)

PLUGIN_NAME = "astrbot_plugin_reply_gate"
_TAG = "[回复判断]"
_PERSONA_CACHE_SECONDS = 60
_SAVE_INTERVAL_SECONDS = 30
_PATH_IDLE = "主动接话"
_PATH_AT = "被@时判断"
_MAX_JUDGE_IMAGES = 3  # 一条消息最多把几张图交给对话模型看

# 所有配置项和默认值；面板保存时只接受这些键
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "persona_aliases": {},
    "extra_names": [],
    "user_blacklist": [],
    "group_whitelist": [],
    "group_blacklist": [],
    "judge_enabled": True,
    "judge_when_at_with_others": True,
    "use_decision_model": True,
    "decision_provider_id": "",
    "threshold": 0.5,
    "open_threshold": 0.6,
    "open_question_chance": 0.5,
    "llm_fallback_enabled": True,
    "image_use_llm": True,
    "llm_provider_id": "",
    "judge_timeout": 8.0,
    "reply_on_error": False,
    "extra_rules": "",
    "context_messages": 8,
    "min_length": 2,
    "cooldown_seconds": 30.0,
    "probability_without_judge": 0.0,
    "log_decisions": True,
    "debug_enabled": False,
    "debug_max_records": 200,
}
_RANGES: dict[str, tuple[float, float]] = {
    "threshold": (0.0, 1.0),
    "open_threshold": (0.0, 1.0),
    "open_question_chance": (0.0, 1.0),
    "probability_without_judge": (0.0, 1.0),
    "judge_timeout": (1.0, 60.0),
    "context_messages": (0, 30),
    "min_length": (0, 50),
    "cooldown_seconds": (0.0, 3600.0),
    "debug_max_records": (10, 2000),
}


def _coerce(key: str, value: Any) -> Any:
    """把面板传来的值转换成配置项应有的类型。转换失败时抛 ValueError。"""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, (int, float)):
        number = float(value)
        low, high = _RANGES.get(key, (float("-inf"), float("inf")))
        number = min(high, max(low, number))
        return int(number) if isinstance(default, int) else number
    if isinstance(default, list):
        return split_names(value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{key} 应该是对象")
        # 值存成「星星，阿星」这样的文本，AstrBot 自带的配置页也能直接编辑
        return {
            str(k).strip(): "，".join(split_names(v))
            for k, v in value.items()
            if str(k).strip() and split_names(v)
        }
    return str(value or "")


def _field(item: Any, *keys: str) -> str:
    """从 dict 或对象里取第一个非空字段。"""
    for key in keys:
        value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if value not in (None, ""):
            return str(value)
    return ""


class _NameLookup:
    """先查这条消息自带的 @ 信息，再查最近发言的人。"""

    def __init__(self, direct: dict[str, str], roster: Roster) -> None:
        self.direct = direct
        self.roster = roster

    def get(self, uid: str, default: str = "") -> str:
        return self.direct.get(uid) or self.roster.name_of(uid) or default


class ReplyGatePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config if config is not None else {}
        self.buffer = ContextBuffer(max_per_group=30)
        self.history: deque[dict[str, Any]] = deque(maxlen=200)
        self.debug: deque[dict[str, Any]] = deque(maxlen=self._debug_max())
        self.roster = Roster()
        self._ids = itertools.count(int(time.time() * 1000))
        self._last_reply: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._persona_cache: dict[str, tuple[float, str | None]] = {}
        self._last_save = 0.0
        self._debug_dirty = False
        self._data_dir = self._resolve_data_dir()
        self._load_state()
        self._register_web_apis()

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = DEFAULTS.get(key)
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _list(self, key: str) -> list[str]:
        return split_names(self._get(key, []))

    def _num(self, key: str, default: float | None = None) -> float:
        if default is None:
            default = float(DEFAULTS.get(key, 0))
        try:
            return float(self._get(key, default))
        except (TypeError, ValueError):
            return default

    def _aliases(self) -> dict[str, Any]:
        value = self._get("persona_aliases", {})
        return value if isinstance(value, dict) else {}

    def _debug_max(self) -> int:
        low, high = _RANGES["debug_max_records"]
        return int(min(high, max(low, self._num("debug_max_records"))))

    def _lock(self, umo: str) -> asyncio.Lock:
        lock = self._locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[umo] = lock
        return lock

    # ------------------------------------------------------------------
    # 持久化：最近出现过的人、群、判断记录和调试记录，重启后还在
    # ------------------------------------------------------------------
    def _resolve_data_dir(self) -> Path | None:
        try:
            from astrbot.api.star import StarTools

            return Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception as exc:
            logger.warning(f"{_TAG} 无法获取数据目录，记录不会保存到磁盘: {exc!r}")
            return None

    def _read_json(self, name: str) -> Any:
        if not self._data_dir:
            return None
        path = self._data_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"{_TAG} 读取 {name} 失败，从空记录开始: {exc!r}")
            return None

    def _write_json(self, name: str, payload: Any) -> None:
        if not self._data_dir:
            return
        path = self._data_dir / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _load_state(self) -> None:
        data = self._read_json("state.json")
        if isinstance(data, dict):
            self.roster.load(data.get("roster"))
            history = data.get("history")
            if isinstance(history, list):
                self.history.extend(i for i in history if isinstance(i, dict))
            self.buffer.load(data.get("context"))
        debug = self._read_json("debug.json")
        if isinstance(debug, list):
            self.debug.extend(i for i in debug if isinstance(i, dict))

    def _save_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_save < _SAVE_INTERVAL_SECONDS:
            return
        self._last_save = now
        try:
            self._write_json(
                "state.json",
                {
                    "roster": self.roster.to_dict(),
                    "history": list(self.history),
                    "context": self.buffer.to_dict(),
                },
            )
            self.roster.dirty = False
            if self._debug_dirty or force:
                self._write_json("debug.json", list(self.debug))
                self._debug_dirty = False
        except Exception as exc:
            logger.warning(f"{_TAG} 保存记录失败: {exc!r}")

    async def terminate(self) -> None:
        self._save_state(force=True)

    # ------------------------------------------------------------------
    # 重启后恢复群会话场景
    #
    # qq_official 适配器只在内存里记「这个会话是 QQ 群还是频道」，要等群里来一条
    # 消息才会记上。重启后这份记录是空的，在此之前往群里主动发的消息都会被
    # 适配器跳过（No cached msg_id ... skip send_by_session）。插件启动时用
    # roster 里记过的群把它补回去。只在启动时跑一次，只补缓存里没有的群。
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        # 插件热重载时平台早已加载好；正常启动时这里平台还没加载，什么都不会做
        self._restore_group_scenes()

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        self._restore_group_scenes()

    def _platform_insts(self) -> list[Any]:
        manager = getattr(self.context, "platform_manager", None)
        return list(getattr(manager, "platform_insts", None) or [])

    def _note_scene(self, event: AstrMessageEvent, group_id: str) -> None:
        """把适配器刚给这个群记下的场景存进 roster，下次重启时用。"""
        if not group_id:
            return
        platform_id = str(event.get_platform_id() or "")
        for inst in self._platform_insts():
            try:
                if inst.meta().id != platform_id:
                    continue
            except Exception:
                continue
            scenes = getattr(inst, "_session_scene", None)
            if isinstance(scenes, dict) and scenes.get(group_id):
                self.roster.note_scene(group_id, str(scenes[group_id]))
            return

    def _restore_group_scenes(self) -> int:
        restored = 0
        for inst in self._platform_insts():
            remember = getattr(inst, "remember_session_scene", None)
            scenes = getattr(inst, "_session_scene", None)
            if not callable(remember) or not isinstance(scenes, dict):
                continue  # 不是 qq_official 这类需要场景缓存的平台
            try:
                meta = inst.meta()
                platform_id, platform_type = str(meta.id), str(meta.name)
            except Exception:
                continue
            prefix = f"{platform_id}:GroupMessage:"
            for group_id, group in list(self.roster.groups.items()):
                umo = str(group.get("umo") or "")
                if not umo.startswith(prefix) or scenes.get(group_id):
                    continue
                scene = str(group.get("scene") or "") or _infer_scene(
                    platform_type, group_id
                )
                if not scene:
                    continue
                try:
                    remember(group_id, scene)
                    restored += 1
                except Exception as exc:
                    logger.warning(f"{_TAG} 恢复群会话场景失败 {group_id}: {exc!r}")
        if restored:
            logger.info(f"{_TAG} 已恢复 {restored} 个群的会话场景，重启后可直接往群里发消息")
        return restored

    # ------------------------------------------------------------------
    # 机器人的名字：跟着会话生效的人格走
    # ------------------------------------------------------------------
    async def _persona_name(self, event: AstrMessageEvent) -> str | None:
        umo = event.unified_msg_origin
        cached = self._persona_cache.get(umo)
        if cached and time.time() - cached[0] < _PERSONA_CACHE_SECONDS:
            return cached[1]
        name: str | None = None
        try:
            conversation_persona = None
            manager = self.context.conversation_manager
            cid = await manager.get_curr_conversation_id(umo)
            if cid:
                conversation = await manager.get_conversation(umo, cid)
                if conversation is not None:
                    conversation_persona = conversation.persona_id or None
            persona_id, persona, _, _ = (
                await self.context.persona_manager.resolve_selected_persona(
                    umo=umo,
                    conversation_persona_id=conversation_persona,
                    platform_name=event.get_platform_name(),
                )
            )
            if persona:
                name = str(persona.get("name") or persona_id or "") or None
        except Exception as exc:
            logger.debug(f"{_TAG} 获取会话人格失败: {exc!r}")
        self._persona_cache[umo] = (time.time(), name)
        return name

    def _names_for(self, persona_name: str | None) -> list[str]:
        names = compose_names(persona_name, self._aliases(), self._list("extra_names"))
        return names or ["机器人"]

    # ------------------------------------------------------------------
    # @ 信息：把 <@openid> 换成昵称，判断模型才知道是对谁说的
    # ------------------------------------------------------------------
    def _read_mentions(
        self, event: AstrMessageEvent, bot_name: str
    ) -> tuple[str, list[str], bool]:
        """返回 (判断用的文字, 被 @ 的其他人, 是否 @ 了机器人)。"""
        message = getattr(event, "message_obj", None)
        raw = getattr(message, "raw_message", None)
        self_id = str(event.get_self_id() or "")
        bot_ids = {self_id} if self_id else set()

        direct: dict[str, str] = {}
        raw_data = getattr(raw, "raw_data", None)
        items = raw_data.get("mentions") if isinstance(raw_data, dict) else None
        if not isinstance(items, list):
            items = getattr(raw, "mentions", None) or []
        for item in items:
            # 同一个人可能同时带几种 ID 字段，文字里用的是哪一种不确定，全部登记
            uids = {
                _field(item, key)
                for key in ("member_openid", "id", "user_openid", "union_openid")
            } - {""}
            if not uids:
                continue
            if _field(item, "is_you") in ("True", "true", "1"):
                bot_ids.update(uids)
                continue
            name = _field(item, "username", "nickname", "nick", "member_name", "name")
            for uid in uids:
                direct[uid] = name

        # 原始文字里还保留着 @ 机器人的标记，优先用它，这样 @ 的位置不会丢
        text = event.message_str or ""
        raw_text = getattr(raw, "content", None)
        if isinstance(raw_text, str) and _MENTION_RE.search(raw_text):
            text = raw_text.strip()
        text, mentioned, at_bot = normalize_mentions(
            text, _NameLookup(direct, self.roster), bot_ids, bot_name
        )

        # 其它平台的 @ 是消息组件，不在文字里
        for comp in event.get_messages():
            if type(comp).__name__ != "At":
                continue
            qq = str(getattr(comp, "qq", "") or "")
            if not qq or qq == "all":
                continue
            if qq in bot_ids or qq == "qq_official":
                at_bot = True
                continue
            name = clean_name(getattr(comp, "name", "")) or clean_name(self.roster.name_of(qq))
            name = name or UNKNOWN_MEMBER
            if name not in mentioned:
                mentioned.append(name)
        return text.strip(), mentioned, at_bot

    def _read_quote(self, event: AstrMessageEvent) -> dict[str, str] | None:
        """这条消息引用了哪句话、是谁说的。认不出说话人时写「不知道是谁」。"""
        for comp in event.get_messages():
            if not isinstance(comp, Reply):
                continue
            quoted = clean_name(
                getattr(comp, "message_str", "") or getattr(comp, "text", "") or ""
            )
            sender = str(getattr(comp, "sender_id", "") or "")
            self_id = str(event.get_self_id() or "")
            if sender and self_id and sender == self_id:
                speaker = BOT_SPEAKER
            else:
                speaker = clean_name(getattr(comp, "sender_nickname", "")) or (
                    self.roster.name_of(sender) if sender else ""
                )
            if not speaker and quoted:
                line = self.buffer.find_quoted(event.unified_msg_origin, quoted)
                if line is not None:
                    speaker = BOT_SPEAKER if line.is_bot else line.speaker
            if quoted or speaker:
                return {"speaker": speaker or "不知道是谁", "text": quoted}
        return None

    # ------------------------------------------------------------------
    # 消息入口
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """群消息：记录上下文，并判断要不要接话。"""
        if not self._get("enabled"):
            return

        umo = event.unified_msg_origin
        sender_id = str(event.get_sender_id() or "")
        group_id = str(event.get_group_id() or "")
        speaker = clean_name(event.get_sender_name()) or f"用户{sender_id[:6]}"
        is_self = bool(sender_id) and sender_id == str(event.get_self_id() or "")
        if not is_self:
            self.roster.see(sender_id, speaker, group_id, umo)
            self._note_scene(event, group_id)

        persona_name = await self._persona_name(event)
        names = self._names_for(persona_name)
        text, mentions, at_bot = self._read_mentions(event, names[0])
        images = [c for c in event.get_messages() if isinstance(c, Image)]
        text = with_image_mark(text, len(images))
        quote = None if is_self else self._read_quote(event)

        # 先取出「这条消息之前」的上下文，再把这条消息记进去
        recent = self.buffer.recent(umo, int(self._num("context_messages")))
        self.buffer.add(umo, speaker, text, is_bot=is_self)
        group = self.roster.groups.get(group_id)
        if group is not None:
            group["persona"] = persona_name or ""
        if self.roster.dirty:
            self._save_state()
        if is_self:
            return

        info = {
            "speaker": speaker,
            "sender_id": sender_id,
            "raw_text": event.message_str or "",
            "judge_text": text,
            "mentions": mentions,
            "at_bot": at_bot,
            "quote": quote,
            "images": images,
            "names": names,
            "persona": persona_name or "",
        }

        if event.is_at_or_wake_command:
            # 被 @ 或被唤醒：照常回复。只有「@ 了机器人、同时也 @ 了别人」时才判断
            if not self._should_judge_at(event, text, mentions, at_bot):
                return
            trace: dict[str, Any] = {}
            try:
                verdict = await self._judge_at(event, info, recent, trace)
            except Exception as exc:
                logger.error(f"{_TAG} 判断过程出错，按照常回复处理: {exc!r}")
                return
            self._record(event, _PATH_AT, info, verdict, trace)
            if not verdict.reply:
                event.should_call_llm(True)  # 只拦 AstrBot 默认回复，不影响其它插件
            return

        if event.get_extra("provider_request") is not None or event.get_result():
            return  # 已经有别的插件决定回复了

        trace = {}
        try:
            verdict = await self._decide(event, info, recent, trace)
        except Exception as exc:  # 判断出错绝不能影响正常聊天
            logger.error(f"{_TAG} 判断过程出错，按不回复处理: {exc!r}")
            return

        self._record(event, _PATH_IDLE, info, verdict, trace)
        if not verdict.reply:
            return

        self._last_reply[umo] = time.time()
        request = await self._build_request(event)
        if request is not None:
            yield request

    _BOT_LINE_FLAG = "_reply_gate_bot_line_recorded"

    def _remember_bot_line(self, event: AstrMessageEvent, text: str) -> None:
        """把机器人说的话记进上下文。同一个事件只记一次。"""
        text = (text or "").strip()
        if not text or not event.get_group_id() or event.get_extra(self._BOT_LINE_FLAG):
            return
        sender_id = str(event.get_sender_id() or "")
        to = clean_name(event.get_sender_name()) or (f"用户{sender_id[:6]}" if sender_id else "")
        self.buffer.add(event.unified_msg_origin, BOT_SPEAKER, text, is_bot=True, reply_to=to)
        event.set_extra(self._BOT_LINE_FLAG, True)
        self._save_state()

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """对话模型的回复：判断「是不是在接它的话」时要用。"""
        try:
            self._remember_bot_line(event, getattr(response, "completion_text", "") or "")
        except Exception as exc:
            logger.debug(f"{_TAG} 记录机器人回复失败: {exc!r}")

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """不经过对话模型发出去的消息（指令结果、其它插件的回复）也记进上下文。"""
        try:
            result = event.get_result()
            if result is not None:
                self._remember_bot_line(event, result.get_plain_text())
        except Exception as exc:
            logger.debug(f"{_TAG} 记录机器人发出的消息失败: {exc!r}")

    # ------------------------------------------------------------------
    # 判断
    # ------------------------------------------------------------------
    def _should_judge_at(
        self, event: AstrMessageEvent, text: str, mentions: list[str], at_bot: bool
    ) -> bool:
        if not (self._get("judge_when_at_with_others") and self._get("judge_enabled")):
            return False
        if not mentions or not at_bot:
            return False
        if event.get_extra("handlers_parsed_params"):
            return False  # 是指令
        plain = _MENTION_RE.sub("", text).strip()
        plain = plain.lstrip("@").strip()
        return not (event.message_str or "").strip().startswith("/") and bool(plain)

    async def _judge_at(
        self, event: AstrMessageEvent, info: dict, recent: list, trace: dict
    ) -> Verdict:
        """@ 了机器人、同时也 @ 了别人：判断是不是真的对机器人说的。出错时照常回复。"""
        names = info["names"]
        state = build_state(
            names, recent, info["speaker"], info["judge_text"], True,
            mentions=info["mentions"], at_bot=True, images=len(info["images"]),
        )
        question = build_at_question(names, str(self._get("extra_rules")))
        trace.update(state=state, question=question)
        probs, stage, note = await self._ask_models(
            event.unified_msg_origin, {"reply": question}, state, trace,
            images=info["images"],
        )
        return self._at_verdict(probs, stage, note)

    def _at_verdict(self, probs: dict | None, stage: str, note: str) -> Verdict:
        if probs is None:
            return Verdict(True, "判断失败", f"{note}；被 @ 时判断失败按照常回复")
        probability = float(probs["reply"])
        threshold = self._num("threshold")
        detail = f"概率 {probability:.2f}，阈值 {threshold:.2f}"
        if note:
            detail += f"；{note}"
        return Verdict(probability >= threshold, stage, detail, probability=probability)

    async def _decide(
        self, event: AstrMessageEvent, info: dict, recent: list, trace: dict
    ) -> Verdict:
        umo = event.unified_msg_origin
        names = info["names"]
        text = info["judge_text"]
        named = mentions_name(text, names)
        info["named"] = named
        cooldown = self._num("cooldown_seconds")

        def since_last() -> float | None:
            last = self._last_reply.get(umo)
            return None if last is None else time.time() - last

        def run_prefilter(use_cooldown: bool) -> Verdict | None:
            return prefilter(
                text=text,
                sender_id=info["sender_id"],
                group_id=str(event.get_group_id() or ""),
                umo=umo,
                named=named,
                user_blacklist=self._list("user_blacklist"),
                group_whitelist=self._list("group_whitelist"),
                group_blacklist=self._list("group_blacklist"),
                min_length=int(self._num("min_length")),
                seconds_since_last_reply=since_last(),
                cooldown_seconds=cooldown if use_cooldown else 0,
                has_image=bool(info["images"]),
            )

        judge = bool(self._get("judge_enabled"))
        # 交给模型判断时，冷却只限制开放问题和新话题；对机器人说的话（比如追问）不受冷却限制
        verdict = run_prefilter(use_cooldown=not judge)
        if verdict is not None:
            return verdict

        if not judge:
            if named:
                return Verdict(True, "叫了名字", "未开启模型判断")
            chance = self._num("probability_without_judge")
            return Verdict(random.random() < chance, "随机概率", f"概率 {chance:g}")

        # 同一个群的判断排队进行，避免几条消息同时判断后一起插嘴
        async with self._lock(umo):
            state = build_state(
                names, recent, info["speaker"], text, named,
                mentions=info["mentions"], quote=info.get("quote"), now=time.time(),
                images=len(info["images"]),
            )
            questions = build_questions(names, str(self._get("extra_rules")))
            trace.update(state=state, question=questions_text(questions))
            probs, stage, note = await self._ask_models(
                umo, questions, state, trace, images=info["images"]
            )
            if probs is None:
                return Verdict(bool(self._get("reply_on_error")), "判断失败", note)
            scores = Scores.of(probs)
            trace["scores"] = scores.to_dict()
            # 排队期间可能已经进入冷却，所以在这里才算
            elapsed = since_last()
            left = cooldown - elapsed if elapsed is not None and cooldown > 0 else 0.0
            verdict = combine(
                scores,
                self._num("threshold"),
                self._num("open_question_chance"),
                random.random(),
                cooldown_left=max(0.0, left),
                open_threshold=self._num("open_threshold"),
            )
            verdict.detail = _with_model_note(verdict.detail, stage, note)
            if verdict.reply:
                self._last_reply[umo] = time.time()
            return verdict

    async def _ask_models(
        self,
        umo: str,
        questions: dict[str, str],
        state: dict,
        trace: dict | None = None,
        images: list | None = None,
    ) -> tuple[dict[str, float] | None, str, str]:
        """问模型。一般先问判断模型，不行再问对话模型。

        判断模型看不了图：消息带图并且开了「图片交给对话模型」时，先问对话模型
        （把图一起发过去），对话模型不行再退回判断模型，只看文字。

        Returns:
            (每个问题为真的概率, 用的哪个模型, 备注)。都失败时概率为 None，备注是失败原因。
        """
        trace = trace if trace is not None else {}
        timeout = self._num("judge_timeout")
        failures: list[str] = []
        started = time.time()
        images = list(images or [])[:_MAX_JUDGE_IMAGES]

        def done(probs: dict[str, float] | None, stage: str, note: str):
            trace["elapsed_ms"] = int((time.time() - started) * 1000)
            trace["failures"] = failures
            if probs is not None:
                trace["probabilities"] = {k: round(v, 3) for k, v in probs.items()}
            return probs, stage, note

        def model_label(kind: str) -> str:
            if kind == "decision":
                return "判断模型 " + (str(self._get("decision_provider_id") or "") or "（默认）")
            label = "对话模型 " + (str(self._get("llm_provider_id") or "") or "（当前会话）")
            return label + (f"，带 {len(images)} 张图" if images else "")

        use_decision = bool(self._get("use_decision_model"))
        use_llm = bool(self._get("llm_fallback_enabled"))
        image_first = bool(images) and use_llm and bool(self._get("image_use_llm"))
        order = ["llm", "decision"] if image_first else ["decision", "llm"]
        if not image_first:
            images = []  # 不走「图片交给对话模型」时，对话模型也只看文字

        for kind in order:
            if kind == "decision" and use_decision:
                trace["model"] = model_label("decision")
                try:
                    probs = await asyncio.wait_for(
                        self._ask_decision(questions, state), timeout
                    )
                    note = "；".join(failures[:1])
                    if image_first:
                        note = "；".join(x for x in ["判断模型看不到图片，只按文字判断", note] if x)
                    return done(probs, "判断模型", note)
                except asyncio.TimeoutError:
                    failures.append("判断模型超时")
                except DecisionError as exc:
                    failures.append(f"判断模型不可用: {exc}")
                except Exception as exc:
                    failures.append(f"判断模型出错: {exc!r}")
            elif kind == "llm" and use_llm:
                trace["model"] = model_label("llm")
                try:
                    probs, reason = await asyncio.wait_for(
                        self._ask_llm(umo, questions, state, trace, images), timeout
                    )
                    if probs is not None:
                        note = "；".join(x for x in [reason, *failures[:1]] if x)
                        return done(probs, "对话模型", note)
                    failures.append(reason or "对话模型的回答看不懂")
                except asyncio.TimeoutError:
                    failures.append("对话模型超时")
                except Exception as exc:
                    failures.append(f"对话模型出错: {exc!r}")

        reason = "；".join(failures) or "没有启用任何模型"
        logger.warning(f"{_TAG} 模型判断失败：{reason}")
        return done(None, "判断失败", reason)

    async def _ask_decision(
        self, questions: dict[str, str], state: dict
    ) -> dict[str, float]:
        provider_id = str(self._get("decision_provider_id") or "").strip() or None
        answers = await self.context.decision.ask(
            {key: DecisionBooleanQuestion(text) for key, text in questions.items()},
            state,
            provider_id=provider_id,
        )
        probs: dict[str, float] = {}
        for key in questions:
            answer = answers.get(key)
            if not isinstance(answer, DecisionBooleanAnswer):
                raise DecisionError("判断模型没有返回是/否结果")
            probs[key] = float(answer.probability)
        return probs

    async def _ask_llm(
        self,
        umo: str,
        questions: dict[str, str],
        state: dict,
        trace: dict | None = None,
        images: list | None = None,
    ) -> tuple[dict[str, float] | None, str]:
        provider_id = str(self._get("llm_provider_id") or "").strip()
        if not provider_id:
            provider = await self.context.get_using_provider_async(umo or None)
            if provider is None:
                raise RuntimeError("没有可用的对话模型")
            provider_id = provider.meta().id
        image_urls: list[str] = []
        for image in images or []:
            try:
                image_urls.append(await image.convert_to_file_path())
            except Exception as exc:
                logger.warning(f"{_TAG} 读取图片失败，判断时跳过这张图: {exc!r}")
        if images and not image_urls:
            raise RuntimeError("图片都读取失败")
        if image_urls and trace is not None:
            trace["images_sent"] = len(image_urls)
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=build_llm_prompt(questions, state),
            image_urls=image_urls or None,
            system_prompt=LLM_SYSTEM_PROMPT,
        )
        raw = getattr(response, "completion_text", "") or ""
        if trace is not None:
            trace["llm_raw"] = raw[:500]
        probs, reason = parse_llm_scores(raw, list(questions))
        if probs is None:
            logger.warning(f"{_TAG} {reason}")
        return probs, reason

    # ------------------------------------------------------------------
    # 发起回复
    # ------------------------------------------------------------------
    async def _build_request(self, event: AstrMessageEvent):
        """和 AstrBot 内置主动回复一样，走正常的 LLM 链路。"""
        umo = event.unified_msg_origin
        manager = self.context.conversation_manager
        try:
            conversation_id = await manager.get_curr_conversation_id(umo)
            if not conversation_id:
                conversation_id = await manager.new_conversation(
                    umo, event.get_platform_id()
                )
            conversation = await manager.get_conversation(umo, conversation_id)
            if not conversation:
                logger.error(f"{_TAG} 找不到对话，无法主动回复")
                return None

            image_urls: list[str] = []
            for component in event.get_messages():
                if isinstance(component, Image):
                    try:
                        image_urls.append(await component.convert_to_file_path())
                    except Exception:
                        logger.warning(f"{_TAG} 处理图片失败，已跳过这张图")

            return event.request_llm(
                prompt=event.message_str,
                session_id=event.session_id,
                image_urls=image_urls,
                conversation=conversation,
            )
        except Exception as exc:
            logger.error(f"{_TAG} 发起主动回复失败: {exc!r}")
            return None

    # ------------------------------------------------------------------
    # 记录与指令
    # ------------------------------------------------------------------
    def _record(
        self,
        event: AstrMessageEvent,
        path: str,
        info: dict,
        verdict: Verdict,
        trace: dict,
    ) -> None:
        text = info["judge_text"]
        short = text if len(text) <= 60 else text[:59] + "…"
        record_id = next(self._ids)
        debug_on = bool(self._get("debug_enabled"))
        self.history.append(
            {
                "id": record_id,
                "ts": time.time(),
                "time": time.strftime("%m-%d %H:%M:%S"),
                "path": path,
                "group_id": str(event.get_group_id() or ""),
                "persona": info["persona"],
                "speaker": info["speaker"],
                "sender_id": info["sender_id"],
                "text": short,
                "reply": verdict.reply,
                "stage": verdict.stage,
                "detail": verdict.detail,
                "probability": verdict.probability,
                "scores": trace.get("scores"),
                "context_count": self._context_count(trace),
                "verdict": verdict.describe(),
                "has_debug": debug_on,
            }
        )
        if debug_on:
            state = trace.get("state") or {}
            self.debug.append(
                {
                    "id": record_id,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "path": path,
                    "group_id": str(event.get_group_id() or ""),
                    "umo": event.unified_msg_origin,
                    "persona": info["persona"],
                    "speaker": info["speaker"],
                    "sender_id": info["sender_id"],
                    "raw_text": info["raw_text"],
                    "judge_text": info["judge_text"],
                    "mentions": info["mentions"],
                    "at_bot": info["at_bot"],
                    "quote": info.get("quote"),
                    "images": len(info.get("images") or []),
                    "images_sent": trace.get("images_sent"),
                    "names": info["names"],
                    "named": info.get("named"),
                    "context": state.get("recent_messages"),
                    "model_input": state or None,
                    "question": trace.get("question"),
                    "model": trace.get("model"),
                    "llm_raw": trace.get("llm_raw"),
                    "failures": trace.get("failures") or [],
                    "elapsed_ms": trace.get("elapsed_ms"),
                    "threshold": self._num("threshold"),
                    "scores": trace.get("scores"),
                    "probabilities": trace.get("probabilities"),
                    "open_question_chance": self._num("open_question_chance"),
                    "open_threshold": self._num("open_threshold"),
                    "reply": verdict.reply,
                    "stage": verdict.stage,
                    "detail": verdict.detail,
                    "probability": verdict.probability,
                }
            )
            self._debug_dirty = True
        if self._get("log_decisions"):
            mark = "（@ 了机器人）" if path == _PATH_AT else ""
            count = self._context_count(trace)
            ctx_note = f" [前文 {count} 条]" if count is not None else ""
            logger.info(
                f"{_TAG} {info['speaker']}{mark}: {short[:30]} → {verdict.describe()}{ctx_note}"
            )
        self._save_state()

    @staticmethod
    def _context_count(trace: dict) -> int | None:
        """交给模型的前文有几条；没走到模型这一步时为 None。"""
        state = trace.get("state")
        if not isinstance(state, dict):
            return None
        return len(state.get("recent_messages") or [])

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("回复判断")
    async def show_history(self, event: AstrMessageEvent):
        """查看最近 10 次判断结果（仅管理员）。"""
        if not self.history:
            yield event.plain_result("还没有判断记录。")
            return
        lines = ["最近的判断记录："]
        for item in list(self.history)[-10:]:
            lines.append(
                f"{item['time']} {item['speaker']}: {item['text'][:30]} → {item['verdict']}"
            )
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------
    # 面板接口
    # ------------------------------------------------------------------
    def _register_web_apis(self) -> None:
        register = getattr(self.context, "register_web_api", None)
        if register is None:
            return
        routes = [
            ("overview", self.api_overview, ["GET"], "回复判断：概览"),
            ("config", self.api_config, ["GET"], "回复判断：读取配置"),
            ("config/save", self.api_save_config, ["POST"], "回复判断：保存配置"),
            ("people", self.api_people, ["GET"], "回复判断：最近出现的人和群"),
            ("list/toggle", self.api_toggle_list, ["POST"], "回复判断：名单增删"),
            ("history/clear", self.api_clear_history, ["POST"], "回复判断：清空记录"),
            ("debug", self.api_debug, ["GET"], "回复判断：调试记录"),
            ("debug/clear", self.api_clear_debug, ["POST"], "回复判断：清空调试记录"),
            ("debug/export", self.api_export_debug, ["GET"], "回复判断：导出调试记录"),
            ("test", self.api_test, ["POST"], "回复判断：试一试"),
        ]
        for path, handler, methods, desc in routes:
            register(f"/{PLUGIN_NAME}/{path}", handler, methods, desc)

    def _web(self):
        from astrbot.api.web import error_response, json_response, request

        return request, json_response, error_response

    def _current_config(self) -> dict[str, Any]:
        config = {key: self._get(key) for key in DEFAULTS}
        config["persona_aliases"] = {
            k: split_names(v) for k, v in self._aliases().items()
        }
        return config

    def _personas(self) -> list[str]:
        try:
            items = self.context.persona_manager.personas_v3
            names = [str(p.get("name") or "") for p in items]
        except Exception:
            names = []
        return [n for n in names if n and n.lower() != "default"]

    def _providers(self) -> dict[str, Any]:
        decision: list[dict[str, Any]] = []
        chat: list[dict[str, Any]] = []
        default_decision = ""
        try:
            manager = self.context.provider_manager
            current = getattr(manager, "curr_decision_provider_inst", None)
            if current is not None:
                default_decision = str(current.provider_config.get("id") or "")
            for provider in getattr(manager, "decision_provider_insts", []):
                cfg = provider.provider_config
                decision.append(
                    {
                        "id": str(cfg.get("id") or ""),
                        "model": str(cfg.get("model") or ""),
                        "source": str(cfg.get("provider") or cfg.get("type") or ""),
                    }
                )
            for provider in getattr(manager, "provider_insts", []):
                cfg = provider.provider_config
                chat.append({"id": str(cfg.get("id") or ""), "model": str(cfg.get("model") or "")})
        except Exception as exc:
            logger.debug(f"{_TAG} 读取模型列表失败: {exc!r}")
        return {"decision": decision, "chat": chat, "default_decision": default_decision}

    def _warnings(self) -> list[str]:
        warnings: list[str] = []
        try:
            confs = self.context.astrbot_config_mgr.confs
            for conf_id, conf in confs.items():
                active = conf.get("provider_ltm_settings", {}).get("active_reply", {})
                if active.get("enable"):
                    label = "默认配置" if conf_id == "default" else f"配置 {conf_id[:8]}"
                    warnings.append(
                        f"{label}里 AstrBot 内置的「主动回复」还开着，"
                        "用了这份配置的群会绕过本插件随机插嘴。"
                    )
        except Exception:
            pass
        if self._get("use_decision_model") and not self._providers()["decision"]:
            warnings.append("还没有配置判断模型，会直接使用对话模型判断。")
        return warnings

    async def api_overview(self):
        _, json_response, _ = self._web()
        history = list(self.history)
        day_ago = time.time() - 86400
        today = [h for h in history if float(h.get("ts") or 0) >= day_ago]
        stages: dict[str, int] = {}
        for item in today:
            stage = str(item.get("stage") or "")
            stages[stage] = stages.get(stage, 0) + 1
        return json_response(
            {
                "enabled": bool(self._get("enabled")),
                "judge_enabled": bool(self._get("judge_enabled")),
                "use_decision_model": bool(self._get("use_decision_model")),
                "threshold": self._num("threshold"),
                "open_threshold": self._num("open_threshold"),
                "decision_provider_id": str(self._get("decision_provider_id") or ""),
                "debug_enabled": bool(self._get("debug_enabled")),
                "providers": self._providers(),
                "warnings": self._warnings(),
                "stats": {
                    "total": len(today),
                    "replied": sum(1 for h in today if h.get("reply")),
                    "at_blocked": sum(
                        1 for h in today if h.get("path") == _PATH_AT and not h.get("reply")
                    ),
                    "stages": stages,
                },
                "history": history[-100:][::-1],
            }
        )

    async def api_config(self):
        _, json_response, _ = self._web()
        return json_response(
            {
                "config": self._current_config(),
                "personas": self._personas(),
                "providers": self._providers(),
            }
        )

    async def api_save_config(self):
        request, json_response, error_response = self._web()
        payload = await request.json(default={})
        data = payload.get("config") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return error_response("配置格式错误")
        changed: list[str] = []
        try:
            for key, value in data.items():
                if key not in DEFAULTS:
                    continue
                value = _coerce(key, value)
                if self._get(key) != value:
                    self.config[key] = value
                    changed.append(key)
        except (TypeError, ValueError) as exc:
            return error_response(f"配置值不正确: {exc}")
        if changed:
            self._persona_cache.clear()
            if "debug_max_records" in changed:
                self.debug = deque(self.debug, maxlen=self._debug_max())
                self._debug_dirty = True
            try:
                self.config.save_config()
            except Exception as exc:
                return error_response(f"保存配置失败: {exc}", status_code=500)
            logger.info(f"{_TAG} 面板修改了配置: {', '.join(changed)}")
        return json_response({"saved": True, "changed": changed, "config": self._current_config()})

    async def api_people(self):
        _, json_response, _ = self._web()
        blacklist = {x.lower() for x in self._list("user_blacklist")}
        whitelist = {x.lower() for x in self._list("group_whitelist")}
        group_black = {x.lower() for x in self._list("group_blacklist")}
        users = [
            {
                "id": uid,
                "name": info.get("name") or "",
                "count": info.get("count") or 0,
                "last_seen": info.get("last_seen") or 0,
                "groups": info.get("groups") or [],
                "blacklisted": uid.lower() in blacklist,
            }
            for uid, info in self.roster.users.items()
        ]
        users.sort(key=lambda u: u["last_seen"], reverse=True)
        known = {u["id"].lower() for u in users}
        for uid in self._list("user_blacklist"):
            if uid.lower() not in known:  # 名单里有、但最近没出现过的人
                users.append(
                    {"id": uid, "name": "", "count": 0, "last_seen": 0, "groups": [], "blacklisted": True}
                )
        groups = [
            {
                "id": gid,
                "umo": info.get("umo") or "",
                "persona": info.get("persona") or "",
                "count": info.get("count") or 0,
                "last_seen": info.get("last_seen") or 0,
                "whitelisted": gid.lower() in whitelist
                or str(info.get("umo") or "").lower() in whitelist,
                "blacklisted": gid.lower() in group_black
                or str(info.get("umo") or "").lower() in group_black,
            }
            for gid, info in self.roster.groups.items()
        ]
        groups.sort(key=lambda g: g["last_seen"], reverse=True)
        return json_response(
            {
                "users": users,
                "groups": groups,
                "group_whitelist": self._list("group_whitelist"),
                "group_blacklist": self._list("group_blacklist"),
            }
        )

    async def api_toggle_list(self):
        request, json_response, error_response = self._web()
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求格式错误")
        key = {
            "user": "user_blacklist",
            "group_white": "group_whitelist",
            "group_black": "group_blacklist",
        }.get(str(payload.get("list") or ""))
        item = str(payload.get("id") or "").strip()
        if not key or not item:
            return error_response("缺少名单类型或 ID")
        current = self._list(key)
        present = any(x.lower() == item.lower() for x in current)
        want = payload.get("on")
        want = (not present) if want is None else bool(want)
        if want and not present:
            current.append(item)
        elif not want and present:
            current = [x for x in current if x.lower() != item.lower()]
        self.config[key] = current
        try:
            self.config.save_config()
        except Exception as exc:
            return error_response(f"保存失败: {exc}", status_code=500)
        logger.info(f"{_TAG} 面板{'加入' if want else '移出'}{key}: {item}")
        return json_response({"list": key, "items": current, "on": want})

    async def api_clear_history(self):
        _, json_response, _ = self._web()
        self.history.clear()
        self._save_state(force=True)
        return json_response({"cleared": True})

    async def api_debug(self):
        request, json_response, _ = self._web()
        try:
            limit = int(request.query.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(limit, 2000))
        records = list(self.debug)[-limit:][::-1]
        record_id = request.query.get("id")
        if record_id:
            records = [r for r in self.debug if str(r.get("id")) == str(record_id)]
        return json_response(
            {
                "enabled": bool(self._get("debug_enabled")),
                "max_records": self._debug_max(),
                "count": len(self.debug),
                "records": records,
            }
        )

    async def api_clear_debug(self):
        _, json_response, _ = self._web()
        self.debug.clear()
        for item in self.history:
            item["has_debug"] = False
        self._debug_dirty = True
        self._save_state(force=True)
        return json_response({"cleared": True})

    async def api_export_debug(self):
        _, json_response, _ = self._web()
        return json_response(
            {"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"), "records": list(self.debug)}
        )

    async def api_test(self):
        """在面板里试一句话：只跑内容检查和模型判断，不看名单和冷却。"""
        request, json_response, error_response = self._web()
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求格式错误")
        text = str(payload.get("text") or "").strip()
        if not text:
            return error_response("请输入要测试的消息")
        speaker = str(payload.get("speaker") or "").strip() or "群友"
        persona = str(payload.get("persona") or "").strip() or None
        names = self._names_for(persona)
        lines = [
            line
            for line in payload.get("context") or []
            if isinstance(line, dict) and str(line.get("text") or "").strip()
        ]
        # 面板里填的前文当作刚刚发生的对话：每条间隔半分钟，最后一条在 20 秒前
        now = time.time()
        recent: list[ChatLine] = []
        last_person = ""
        for index, line in enumerate(lines):
            is_bot = bool(line.get("is_bot"))
            who = str(line.get("speaker") or "群友")
            recent.append(
                ChatLine(
                    speaker=who,
                    text=str(line.get("text")),
                    is_bot=is_bot,
                    ts=now - 20 - 30 * (len(lines) - 1 - index),
                    reply_to=last_person if is_bot else "",
                )
            )
            if not is_bot:
                last_person = who
        # 文字里写的「@名字」：是机器人的名字就算 @ 了机器人，其余算 @ 了别人
        mentions: list[str] = []
        at_bot = False
        for token in _split_at_tokens(text):
            if token in names:
                at_bot = True
            elif token not in mentions:
                mentions.append(token)
        use_at = at_bot and bool(mentions)
        named = mentions_name(text, names)
        started = time.time()
        trace: dict[str, Any] = {}
        if not use_at and not named and is_noise(text, int(self._num("min_length"))):
            verdict = Verdict(False, "没有实质内容")
        elif use_at:
            state = build_state(
                names, recent, speaker, text, True, mentions=mentions, at_bot=True
            )
            question = build_at_question(names, str(self._get("extra_rules")))
            trace.update(state=state, question=question)
            probs, stage, note = await self._ask_models("", {"reply": question}, state, trace)
            verdict = self._at_verdict(probs, stage, note)
        else:
            state = build_state(
                names, recent, speaker, text, named, mentions=mentions, now=now
            )
            questions = build_questions(names, str(self._get("extra_rules")))
            trace.update(state=state, question=questions_text(questions))
            probs, stage, note = await self._ask_models("", questions, state, trace)
            if probs is None:
                verdict = Verdict(bool(self._get("reply_on_error")), "判断失败", note)
            else:
                scores = Scores.of(probs)
                trace["scores"] = scores.to_dict()
                # 试一试不掷骰子：开放问题、新话题只要接话概率大于 0 就显示「会回复」
                verdict = combine(
                    scores, self._num("threshold"), self._num("open_question_chance"), 0.0,
                    open_threshold=self._num("open_threshold"),
                )
                verdict.detail = _with_model_note(verdict.detail, stage, note)
        return json_response(
            {
                "names": names,
                "named": named,
                "mentions": mentions,
                "path": _PATH_AT if use_at else _PATH_IDLE,
                "reply": verdict.reply,
                "stage": verdict.stage,
                "detail": verdict.detail,
                "probability": verdict.probability,
                "threshold": self._num("threshold"),
                "scores": trace.get("scores"),
                "open_question_chance": self._num("open_question_chance"),
                "open_threshold": self._num("open_threshold"),
                "elapsed_ms": int((time.time() - started) * 1000),
                "model_input": trace.get("state"),
                "question": trace.get("question"),
            }
        )


def _with_model_note(detail: str, stage: str, note: str) -> str:
    """不是判断模型给的结论，或者有备注（比如判断模型看不到图）时，写进详情。"""
    extra = stage if stage != "判断模型" else ""
    if note:
        extra = f"{extra}：{note}" if extra else note
    return f"{detail}；{extra}" if extra else detail


def _infer_scene(platform_type: str, group_id: str) -> str:
    """旧记录没存场景时的兜底：qq_official 的 QQ 群 openid 是 32 位十六进制，
    频道 ID 是纯数字。认不出来就不猜，返回空串。"""
    if platform_type.startswith("qq_official") and len(group_id) == 32:
        try:
            int(group_id, 16)
        except ValueError:
            return ""
        return "group"
    return ""


def _split_at_tokens(text: str) -> list[str]:
    """从「@小明，和@小星 说句话」里取出 ["小明", "小星"]。"""
    import re

    return [m.strip() for m in re.findall(r"@([^\s@，,。.！!？?：:]+)", text or "") if m.strip()]
