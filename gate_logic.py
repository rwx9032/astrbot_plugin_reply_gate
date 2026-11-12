"""回复判断的纯逻辑部分：不依赖 AstrBot，可以单独测试。"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# 整句都是这类内容时，认为没有实质内容，不值得接话
_LAUGH_RE = re.compile(
    r"(哈|呵|嘿|嘻|啊|哦|噢|嗯|额|呃|草|艹|笑死|笑|乐|好的|好|行|ok|okay|h|w|ha|he|lol|xswl|233+|666+|6+)+",
    re.IGNORECASE,
)
# 去掉空白、标点、表情符号之后剩下的才算「字」
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def clean_name(name: Any) -> str:
    """去掉昵称里看不见的字符（方向控制符、零宽字符等），首尾空白也去掉。

    有人的 QQ 昵称就是一个 U+202D，直接显示是一片空白。
    """
    text = str(name or "")
    return "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc", "Co", "Cn")
    ).strip()


def core_text(text: str) -> str:
    """去掉空白、标点和表情后的文本。"""
    return _NON_WORD_RE.sub("", text or "")


def is_noise(text: str, min_length: int = 2) -> bool:
    """这句话是不是没有实质内容（太短、纯笑声、纯表情或标点）。"""
    core = core_text(text)
    if len(core) < max(1, min_length):
        return True
    return _LAUGH_RE.fullmatch(core) is not None


def mentions_name(text: str, names: list[str]) -> bool:
    """消息里有没有出现机器人的名字或别称。"""
    lowered = (text or "").lower()
    return any(name.strip() and name.strip().lower() in lowered for name in names)


def compose_names(
    persona_name: str | None,
    aliases: dict[str, Any] | None,
    extra_names: list[str] | None = None,
) -> list[str]:
    """机器人的名字列表：第一个是人格名，后面是这个人格的别称和全局额外名字。

    人格名为空或是系统默认人格（default）时，没有正式名字，只用额外名字。
    """
    names: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in names:
            names.append(text)

    persona = str(persona_name or "").strip()
    if persona and persona.lower() != "default" and not persona.startswith("["):
        add(persona)
        raw = (aliases or {}).get(persona)
        if isinstance(raw, str):
            raw = re.split(r"[,，、\s]+", raw)
        for alias in raw or []:
            add(alias)
    for name in extra_names or []:
        add(name)
    return names


def split_names(value: Any) -> list[str]:
    """把「星星，阿星、小助手」这类输入拆成名字列表。"""
    if isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        parts = re.split(r"[,，、\n]+", str(value or ""))
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if part and part not in result:
            result.append(part)
    return result


def id_in_list(items: list[str], *candidates: str) -> bool:
    """名单里有没有命中任意一个候选 ID（忽略大小写和首尾空白）。"""
    wanted = {str(item).strip().lower() for item in items if str(item).strip()}
    if not wanted:
        return False
    return any(str(c).strip().lower() in wanted for c in candidates if c)


@dataclass
class ChatLine:
    speaker: str
    text: str
    is_bot: bool = False
    ts: float = field(default_factory=time.time)
    reply_to: str = ""  # 机器人的话：是回给谁的


class ContextBuffer:
    """每个群保留最近若干条消息，给判断模型当上下文。"""

    def __init__(self, max_per_group: int = 30, max_groups: int = 200) -> None:
        self.max_per_group = max_per_group
        self.max_groups = max_groups
        self._data: dict[str, deque[ChatLine]] = {}

    def add(
        self,
        group: str,
        speaker: str,
        text: str,
        is_bot: bool = False,
        reply_to: str = "",
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        lines = self._data.get(group)
        if lines is None:
            if len(self._data) >= self.max_groups:
                # 丢掉最久没有动静的群
                oldest = min(
                    self._data, key=lambda g: self._data[g][-1].ts if self._data[g] else 0
                )
                self._data.pop(oldest, None)
            lines = deque(maxlen=self.max_per_group)
            self._data[group] = lines
        lines.append(ChatLine(speaker=speaker, text=text, is_bot=is_bot, reply_to=reply_to))

    def find_quoted(self, group: str, quoted: str) -> ChatLine | None:
        """被引用的那句话是谁说的：在最近的消息里找内容对得上的一条。"""
        quoted = core_text(quoted)
        if not quoted:
            return None
        for line in reversed(self._data.get(group) or ()):
            mine = core_text(line.text)
            if not mine:
                continue
            if mine == quoted:
                return line
            # 引用里的文字可能被截断，也可能多带了别的内容；太短的不做模糊匹配
            if len(quoted) >= 4 and quoted[:30] in mine:
                return line
            if len(mine) >= 4 and mine[:30] in quoted:
                return line
        return None

    def to_dict(self, max_chars: int = 300) -> dict[str, list[dict[str, Any]]]:
        """存盘用：重启后上下文还在，不用等群里重新攒消息。"""
        return {
            group: [
                {
                    "speaker": line.speaker,
                    "text": line.text[:max_chars],
                    "is_bot": line.is_bot,
                    "ts": line.ts,
                    "reply_to": line.reply_to,
                }
                for line in lines
            ]
            for group, lines in self._data.items()
            if lines
        }

    def load(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        for group, items in data.items():
            if not isinstance(items, list):
                continue
            lines: deque[ChatLine] = deque(maxlen=self.max_per_group)
            for item in items:
                if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                    continue
                try:
                    ts = float(item.get("ts") or 0)
                except (TypeError, ValueError):
                    continue
                lines.append(
                    ChatLine(
                        speaker=clean_name(item.get("speaker")) or "群友",
                        text=str(item["text"]),
                        is_bot=bool(item.get("is_bot")),
                        ts=ts,
                        reply_to=clean_name(item.get("reply_to")),
                    )
                )
            if lines:
                self._data[str(group)] = lines
        while len(self._data) > self.max_groups:
            oldest = min(self._data, key=lambda g: self._data[g][-1].ts)
            self._data.pop(oldest, None)

    def recent(self, group: str, count: int) -> list[ChatLine]:
        lines = self._data.get(group)
        if not lines or count <= 0:
            return []
        return list(lines)[-count:]


class Roster:
    """记住最近在群里出现过的人和群，面板里可以直接从这里拉黑，不用手动找 ID。"""

    def __init__(self, max_users: int = 500, max_groups: int = 100) -> None:
        self.max_users = max_users
        self.max_groups = max_groups
        self.users: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, Any]] = {}
        self.dirty = False

    def see(self, user_id: str, name: str, group_id: str, umo: str) -> None:
        now = time.time()
        if user_id:
            user = self.users.get(user_id)
            if user is None:
                user = {"name": name, "count": 0, "groups": []}
                self.users[user_id] = user
                self.dirty = True
            if name and user.get("name") != name:
                user["name"] = name
                self.dirty = True
            if group_id and group_id not in user["groups"]:
                user["groups"] = (user["groups"] + [group_id])[-5:]
                self.dirty = True
            user["count"] = int(user.get("count") or 0) + 1
            user["last_seen"] = now
            self._trim(self.users, self.max_users)
        if group_id:
            group = self.groups.get(group_id)
            if group is None:
                group = {"umo": umo, "count": 0}
                self.groups[group_id] = group
                self.dirty = True
            group["umo"] = umo or group.get("umo", "")
            group["count"] = int(group.get("count") or 0) + 1
            group["last_seen"] = now
            self._trim(self.groups, self.max_groups)

    def note_scene(self, group_id: str, scene: str) -> None:
        """记下平台适配器给这个群标的场景（qq_official 是 group / channel）。"""
        group = self.groups.get(group_id)
        if group is not None and scene and group.get("scene") != scene:
            group["scene"] = scene
            self.dirty = True

    def _trim(self, table: dict[str, dict[str, Any]], limit: int) -> None:
        while len(table) > limit:
            oldest = min(table, key=lambda k: table[k].get("last_seen", 0))
            table.pop(oldest, None)
            self.dirty = True

    def name_of(self, user_id: str) -> str:
        return str((self.users.get(user_id) or {}).get("name") or "")

    def to_dict(self) -> dict[str, Any]:
        return {"users": self.users, "groups": self.groups}

    def load(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        users = data.get("users")
        groups = data.get("groups")
        if isinstance(users, dict):
            self.users = {
                str(k): v for k, v in users.items() if isinstance(v, dict)
            }
            for user in self.users.values():
                if not isinstance(user.get("groups"), list):
                    user["groups"] = []
                user["name"] = clean_name(user.get("name"))  # 旧记录里可能有看不见的字符
        if isinstance(groups, dict):
            self.groups = {
                str(k): v for k, v in groups.items() if isinstance(v, dict)
            }


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# QQ 官方机器人：<@openid>、<@!openid>、<qqbot-at-user id="openid" />
_MENTION_RE = re.compile(r'<qqbot-at-user\s+id="([^"]+)"\s*/>|<@!?([^>\s]+)>')
UNKNOWN_MEMBER = "某位群友"


def normalize_mentions(
    text: str,
    id_to_name: dict[str, str],
    bot_ids: set[str],
    bot_name: str,
) -> tuple[str, list[str], bool]:
    """把消息里的 @ 标记换成「@昵称」，判断模型才看得懂是对谁说的。

    Returns:
        (换好的文字, 被 @ 的其他人的昵称, 文字里是否 @ 了机器人)
    """
    mentioned: list[str] = []
    at_bot = False

    def replace(match: re.Match) -> str:
        nonlocal at_bot
        uid = match.group(1) or match.group(2) or ""
        if uid in bot_ids:
            at_bot = True
            return f"@{bot_name}"
        name = clean_name(id_to_name.get(uid)) or UNKNOWN_MEMBER
        if name not in mentioned:
            mentioned.append(name)
        return f"@{name}"

    return _MENTION_RE.sub(replace, text or ""), mentioned, at_bot


BOT_SPEAKER = "机器人自己"
IMAGE_MARK = "[图片]"


def with_image_mark(text: str, images: int) -> str:
    """带图的消息在文字后面标上「[图片]」，纯图片消息就只剩这个标记。"""
    text = (text or "").strip()
    if images <= 0:
        return text
    mark = IMAGE_MARK if images == 1 else f"{IMAGE_MARK}×{images}"
    return f"{text} {mark}".strip()


def ago_label(seconds: float) -> str:
    """把「多少秒以前」写成模型容易看懂的说法。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}秒前"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时前"
    return f"{int(seconds // 86400)}天前"


def build_state(
    bot_names: list[str],
    recent: list[ChatLine],
    speaker: str,
    text: str,
    named: bool,
    max_chars: int = 200,
    mentions: list[str] | None = None,
    at_bot: bool = False,
    quote: dict[str, str] | None = None,
    now: float | None = None,
    images: int = 0,
) -> dict[str, Any]:
    """交给判断模型的上下文。

    now 不为空时，每条前文带上「多久以前」，机器人的话带上「回给谁」，
    模型才分得清是在接机器人的话，还是隔了很久的新话题。
    """
    current: dict[str, Any] = {
        "speaker": speaker,
        "text": _clip(text, max_chars),
        "mentions_bot_name": named,
    }
    # 没有 @ 别人时不加这两个字段，保持和调好阈值时的输入一致
    if mentions:
        current["mentions"] = list(mentions)
    if at_bot:
        current["at_bot"] = True
    if images > 0:
        current["images"] = images
    if quote and quote.get("text"):
        current["quoting"] = {
            "speaker": quote.get("speaker") or "不知道是谁",
            "text": _clip(quote["text"], max_chars),
        }

    def line_of(line: ChatLine) -> dict[str, Any]:
        item: dict[str, Any] = {
            "speaker": BOT_SPEAKER if line.is_bot else line.speaker,
            "text": _clip(line.text, max_chars),
        }
        if now is not None:
            item["time"] = ago_label(now - line.ts)
            if line.is_bot and line.reply_to:
                item["replying_to"] = line.reply_to
        return item

    return {
        "bot_name": bot_names[0] if bot_names else "机器人",
        "bot_aliases": bot_names[1:],
        "recent_messages": [line_of(line) for line in recent],
        "current_message": current,
    }


def _bot_label(bot_names: list[str]) -> str:
    name = bot_names[0] if bot_names else "机器人"
    aliases = "、".join(bot_names[1:])
    return f"「{name}」（别称：{aliases}）" if aliases else f"「{name}」"


def build_at_question(bot_names: list[str], extra_rules: str = "") -> str:
    """「@ 了机器人、同时也 @ 了别人」时问判断模型的陈述句：为真表示应该回复。"""
    question = (
        f"群聊机器人{_bot_label(bot_names)}现在应该回应 current_message 这条消息。"
        "这条消息 @ 了机器人，同时也 @ 了别人（见 current_message.mentions）。"
        "如果这句话是在让机器人回答、做事，或者是在和机器人对话，应该回应。"
        "如果这句话其实是对被 @ 的别人说的，只是顺带提到或 @ 了机器人，"
        "例如让别人去和机器人说话、向别人介绍机器人、评论机器人说过的话，不应该回应。"
    )
    extra_rules = (extra_rules or "").strip()
    if extra_rules:
        question += f"另外还要遵守这些规则：{extra_rules}"
    return question


# 没有 @ 机器人的消息，拆成三个更具体的问题分别问，再由 combine() 决定回不回。
# 一个笼统的「要不要回」，小模型很难同时顾及「对谁说」和「是不是对全群的问题」。
Q_TO_BOT = "to_bot"
Q_TO_OTHER = "to_other"
Q_OPEN = "open"
Q_NEW_TOPIC = "new_topic"
QUESTION_LABELS = {
    Q_TO_BOT: "对机器人说",
    Q_TO_OTHER: "对别人说",
    Q_OPEN: "对全群说",
    Q_NEW_TOPIC: "新话题",
}
OPEN_THRESHOLD = 0.6  # 默认：「对全群说」或「新话题」达到这个值才算开放问题 / 新话题
OTHER_BLOCK = 0.4  # 「对别人说」达到这个值就不算开放问题 / 新话题（接着别人聊的话常在 0.45～0.55）


def build_questions(bot_names: list[str], extra_rules: str = "") -> dict[str, str]:
    """没有 @ 机器人时问判断模型的三个陈述句。"""
    to_bot = (
        f"current_message 是说给群聊机器人{_bot_label(bot_names)}的"
        f"（机器人在 recent_messages 里显示为「{BOT_SPEAKER}」）。"
        "例如：叫了机器人的名字；引用（quoting）了机器人的消息；"
        "机器人刚刚回复过这个说话人（replying_to 是他），他接着往下说；"
        "追问机器人刚回答的内容。"
        "如果这句话是在回答或接着别的群成员说的话，或者机器人上次说话已经过去很久，就不是说给机器人的。"
    )
    extra_rules = (extra_rules or "").strip()
    if extra_rules:
        to_bot += f"另外，按这些规则判断（规则说要回应的算是，说不要回应的算不是）：{extra_rules}"
    return {
        Q_TO_BOT: to_bot,
        Q_TO_OTHER: (
            "current_message 是说给某一个特定的群成员（不是机器人）的。"
            "例如：称呼或 @ 了别人；引用了别人的消息；在回答别人刚刚问说话人的问题；"
            "说话人和某个人正在一来一回地聊天，这句接着那段对话。"
        ),
        Q_OPEN: (
            "current_message 是对全群所有人说的：一个开放性的问题、招呼或求助，没有特定的对象，"
            "群里任何人都可以接，例如「吃了吗」「有人知道……吗」「大家觉得……」「大家早上好」。"
            "如果这句话明显是说给某个特定的人或者机器人的，就不算。"
        ),
        Q_NEW_TOPIC: (
            "current_message 开启了一个新话题：它和 recent_messages 里正在聊的内容无关，"
            "或者前面已经很久没人说话了。"
            "并且它的内容是别人可以接着聊的，比如分享了一件事、一个想法、一个问题；"
            "「测试」「嗯」「在吗」这种没法接着聊的话不算。"
            "如果它是在接着前面的话题说、在回应某个人，就不算新话题。"
        ),
    }


def questions_text(questions: dict[str, str]) -> str:
    """调试记录里展示用。"""
    if len(questions) == 1:
        return next(iter(questions.values()))
    return "\n\n".join(f"【{QUESTION_LABELS.get(k, k)}】{q}" for k, q in questions.items())


@dataclass
class Scores:
    to_bot: float
    to_other: float
    open: float
    new_topic: float = 0.0

    @classmethod
    def of(cls, probs: dict[str, float]) -> "Scores":
        return cls(
            float(probs.get(Q_TO_BOT, 0.0)),
            float(probs.get(Q_TO_OTHER, 0.0)),
            float(probs.get(Q_OPEN, 0.0)),
            float(probs.get(Q_NEW_TOPIC, 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            Q_TO_BOT: round(self.to_bot, 3),
            Q_TO_OTHER: round(self.to_other, 3),
            Q_OPEN: round(self.open, 3),
            Q_NEW_TOPIC: round(self.new_topic, 3),
        }

    def describe(self) -> str:
        return (
            f"对机器人 {self.to_bot:.2f} · 对别人 {self.to_other:.2f} · "
            f"对全群 {self.open:.2f} · 新话题 {self.new_topic:.2f}"
        )


def combine(
    scores: Scores,
    threshold: float,
    open_chance: float,
    roll: float,
    cooldown_left: float = 0.0,
    open_threshold: float = OPEN_THRESHOLD,
) -> "Verdict":
    """根据四个分数决定回不回。

    - 对机器人说的（并且比「对别人说」更像）：回复，不受冷却限制。
    - 对全群的开放问题、开启了新话题（都不是对某个人说的）：按 open_chance 的概率接话，
      受冷却限制。
    - 其余（对别人说的、接着别人的话题聊、闲聊）：不回。
    """
    detail = scores.describe()
    if scores.to_bot >= threshold and scores.to_bot >= scores.to_other:
        return Verdict(True, "对机器人说", detail, probability=scores.to_bot)
    stage = ""
    probability = 0.0
    if scores.to_other < OTHER_BLOCK:
        if scores.open >= open_threshold:
            stage, probability = "开放问题", scores.open
        elif scores.new_topic >= open_threshold:
            stage, probability = "新话题", scores.new_topic
    if stage:
        if cooldown_left > 0:
            return Verdict(
                False, "冷却中", f"{stage}，冷却还剩 {int(cooldown_left)} 秒；{detail}",
                probability=probability,
            )
        chance = f"接话概率 {round(open_chance * 100)}%"
        if roll < open_chance:
            return Verdict(True, stage, f"{chance}；{detail}", probability=probability)
        return Verdict(False, stage, f"{chance}，这次没接；{detail}", probability=probability)
    if scores.to_other >= OTHER_BLOCK:
        return Verdict(False, "对别人说", detail, probability=scores.to_bot)
    return Verdict(False, "无需回应", detail, probability=scores.to_bot)


LLM_SYSTEM_PROMPT = (
    "你是一个群聊机器人的「要不要接话」判断器。"
    "你会收到几条陈述和一段 JSON 上下文，要判断每条陈述成立的可能性，用 0 到 1 的数字表示。"
    '只输出一个 JSON 对象，例如 {"陈述的键": 0.8, "reason": "一句话理由"}，'
    "不要输出任何别的内容。"
)


def build_llm_prompt(questions: dict[str, str], state: dict[str, Any]) -> str:
    listed = "\n".join(f"- {key}：{text}" for key, text in questions.items())
    keys = "、".join(questions)
    images = (state.get("current_message") or {}).get("images") or 0
    image_note = (
        f"current_message 附带了 {images} 张图片，已经随这条消息发给你。"
        "判断时要看图片内容，文字里的「[图片]」就是这些图片。\n\n"
        if images
        else ""
    )
    return (
        f"{image_note}陈述：\n{listed}\n\n"
        f"上下文：\n{json.dumps(state, ensure_ascii=False, indent=1)}\n\n"
        f"请输出 JSON，包含 {keys} 和 reason。"
    )


def _as_probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("true", "yes", "是"):
            return 1.0
        if word in ("false", "no", "否"):
            return 0.0
        try:
            return min(1.0, max(0.0, float(word)))
        except ValueError:
            return None
    return None


def parse_llm_scores(text: str, keys: list[str]) -> tuple[dict[str, float] | None, str]:
    """解析对话模型的回答。看不懂时返回 (None, 原因)。"""
    raw = (text or "").strip()
    if not raw:
        return None, "模型返回为空"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            probs = {k: _as_probability(data.get(k)) for k in keys}
            reason = str(data.get("reason") or "").strip()
            if all(v is not None for v in probs.values()):
                return probs, reason  # type: ignore[return-value]
            # 漏答了几个：没答的当作 0（只会让机器人更安静），至少要答出一个
            if any(v is not None for v in probs.values()):
                missing = "、".join(k for k, v in probs.items() if v is None)
                filled = {k: (0.0 if v is None else v) for k, v in probs.items()}
                return filled, "；".join(x for x in [reason, f"模型没回答 {missing}，按 0 算"] if x)
    loose: dict[str, float] = {}
    for key in keys:
        found = re.search(
            rf'"?{re.escape(key)}"?\s*[:：]\s*"?(true|false|[01](?:\.\d+)?)', raw, re.IGNORECASE
        )
        if found:
            loose[key] = _as_probability(found.group(1)) or 0.0
    if len(loose) == len(keys):
        return loose, ""
    return None, f"看不懂模型的回答：{_clip(raw, 60)}"


@dataclass
class Verdict:
    reply: bool
    stage: str  # 在哪一步做出的决定
    detail: str = ""  # 理由或概率
    probability: float | None = None  # 判断模型给出的概率

    def describe(self) -> str:
        head = "回复" if self.reply else "不回"
        tail = f"，{self.detail}" if self.detail else ""
        return f"{head}（{self.stage}{tail}）"


def prefilter(
    *,
    text: str,
    sender_id: str,
    group_id: str,
    umo: str,
    named: bool,
    user_blacklist: list[str],
    group_whitelist: list[str],
    group_blacklist: list[str],
    min_length: int,
    seconds_since_last_reply: float | None,
    cooldown_seconds: float,
    has_image: bool = False,
) -> Verdict | None:
    """不调用模型就能下结论的规则。返回 None 表示需要交给模型判断。

    带图的消息不做「没有实质内容」检查：文字再少，图片本身也可能需要回应。
    """
    if group_whitelist and not id_in_list(group_whitelist, group_id, umo):
        return Verdict(False, "群不在白名单")
    if id_in_list(group_blacklist, group_id, umo):
        return Verdict(False, "群在黑名单")
    if id_in_list(user_blacklist, sender_id):
        return Verdict(False, "用户黑名单")
    if named:
        # 叫了名字：跳过内容和冷却检查，直接交给模型确认
        return None
    if not has_image and is_noise(text, min_length):
        return Verdict(False, "没有实质内容")
    if (
        cooldown_seconds > 0
        and seconds_since_last_reply is not None
        and seconds_since_last_reply < cooldown_seconds
    ):
        left = int(cooldown_seconds - seconds_since_last_reply)
        return Verdict(False, "冷却中", f"还剩 {left} 秒")
    return None
