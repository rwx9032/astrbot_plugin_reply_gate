"""gate_logic 的离线测试：python3 -m pytest tests 或直接运行本文件。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_logic import (  # noqa: E402
    ChatLine,
    ContextBuffer,
    Roster,
    Scores,
    ago_label,
    build_at_question,
    build_llm_prompt,
    build_questions,
    build_state,
    combine,
    compose_names,
    id_in_list,
    is_noise,
    mentions_name,
    normalize_mentions,
    parse_llm_scores,
    prefilter,
    questions_text,
    split_names,
    with_image_mark,
)

OPENID = "0123456789ABCDEF0123456789ABCDEF"


def _pre(**overrides):
    args = dict(
        text="今天店里生意怎么样",
        sender_id="USER1",
        group_id="GROUP1",
        umo="bot:GroupMessage:GROUP1",
        named=False,
        user_blacklist=[],
        group_whitelist=[],
        group_blacklist=[],
        min_length=2,
        seconds_since_last_reply=None,
        cooldown_seconds=30,
    )
    args.update(overrides)
    return prefilter(**args)


def test_noise():
    assert is_noise("哈哈哈哈哈哈哈哈")
    assert is_noise("hhhh")
    assert is_noise("？？？")
    assert is_noise("😂😂")
    assert is_noise("6")
    assert is_noise("  ")
    assert is_noise("好的")
    assert not is_noise("小明，你作业写完了吗")
    assert not is_noise("今天吃什么")
    assert not is_noise("哈哈你说得对")


def test_names():
    assert mentions_name("星星在吗", ["小星", "星星"])
    assert not mentions_name("小明", ["小星", "星星"])
    assert not mentions_name("随便说点什么", ["", " "])


def test_compose_names_follows_persona():
    aliases = {"小星": "星星，管家", "别人": ["阿别"]}
    assert compose_names("小星", aliases) == ["小星", "星星", "管家"]
    assert compose_names("别人", aliases, ["机器人"]) == ["别人", "阿别", "机器人"]
    # 换了人格，名字跟着换，旧人格的别称不会带过来
    assert compose_names("新人格", aliases) == ["新人格"]
    # 系统默认人格、没有人格、显式不用人格：没有正式名字
    assert compose_names("default", aliases, ["小助手"]) == ["小助手"]
    assert compose_names(None, aliases) == []
    assert compose_names("[%None]", aliases) == []
    # 去重
    assert compose_names("小星", {"小星": "小星，星星"}, ["星星"]) == ["小星", "星星"]


def test_split_names():
    assert split_names("星星，阿星, 管家、 老板\n店主") == ["星星", "阿星", "管家", "老板", "店主"]
    assert split_names(["a", " a ", "", "b"]) == ["a", "b"]
    assert split_names(None) == []


def test_roster():
    roster = Roster(max_users=2, max_groups=5)
    roster.see("U1", "甲", "G1", "bot:GroupMessage:G1")
    assert roster.dirty and roster.users["U1"]["name"] == "甲"
    roster.dirty = False
    roster.see("U1", "甲", "G1", "bot:GroupMessage:G1")
    assert not roster.dirty and roster.users["U1"]["count"] == 2  # 只是计数，不触发保存
    roster.see("U1", "甲改名", "G1", "")
    assert roster.dirty and roster.name_of("U1") == "甲改名"
    roster.users["U1"]["last_seen"] = 1  # 让 U1 变成最久没出现的
    roster.see("U2", "乙", "G1", "")
    roster.see("U3", "丙", "G2", "")
    assert set(roster.users) == {"U2", "U3"}
    copy = Roster()
    copy.load(roster.to_dict())
    assert copy.users["U3"]["groups"] == ["G2"] and "G2" in copy.groups
    copy.load({"users": {"X": "坏数据"}, "groups": None})
    assert copy.users == {}


def test_id_match():
    assert id_in_list([OPENID], OPENID.lower())
    assert id_in_list([f" {OPENID} "], OPENID)
    assert not id_in_list([OPENID], "OTHER")
    assert not id_in_list([], OPENID)
    assert not id_in_list([OPENID], "")


def test_prefilter_blacklist_beats_name():
    verdict = _pre(sender_id=OPENID, user_blacklist=[OPENID], named=True, text="星星你好")
    assert verdict is not None and not verdict.reply and verdict.stage == "用户黑名单"


def test_prefilter_groups():
    assert _pre(group_whitelist=["OTHER"]).stage == "群不在白名单"
    assert _pre(group_whitelist=["GROUP1"]) is None
    assert _pre(group_whitelist=["bot:GroupMessage:GROUP1"]) is None
    assert _pre(group_blacklist=["GROUP1"]).stage == "群在黑名单"


def test_prefilter_noise_and_cooldown():
    assert _pre(text="哈哈哈哈").stage == "没有实质内容"
    assert _pre(seconds_since_last_reply=5).stage == "冷却中"
    assert _pre(seconds_since_last_reply=31) is None
    assert _pre(seconds_since_last_reply=5, cooldown_seconds=0) is None
    # 叫了名字：跳过内容和冷却检查
    assert _pre(text="小星", named=True, seconds_since_last_reply=1) is None


def test_images():
    assert with_image_mark("", 1) == "[图片]"
    assert with_image_mark("看这个", 2) == "看这个 [图片]×2"
    assert with_image_mark("看这个", 0) == "看这个"
    # 纯图片、图加一个字：不当作没有实质内容
    assert _pre(text="[图片]", has_image=True) is None
    assert _pre(text="6 [图片]", has_image=True) is None
    assert _pre(text="6").stage == "没有实质内容"
    state = build_state(["小星"], [], "甲", "[图片]", False, images=1)
    assert state["current_message"]["images"] == 1
    assert "附带了 1 张图片" in build_llm_prompt({"to_bot": "x"}, state)
    assert "图片" not in build_llm_prompt({"to_bot": "x"}, build_state(["小星"], [], "甲", "嗨", False))


def test_prefilter_normal_goes_to_model():
    assert _pre() is None


def test_buffer():
    buf = ContextBuffer(max_per_group=3, max_groups=2)
    for i in range(5):
        buf.add("g1", "甲", f"消息{i}")
    assert [line.text for line in buf.recent("g1", 10)] == ["消息2", "消息3", "消息4"]
    assert [line.text for line in buf.recent("g1", 2)] == ["消息3", "消息4"]
    buf.add("g1", "乙", "   ")  # 空消息不记录
    assert len(buf.recent("g1", 10)) == 3
    buf.add("g2", "甲", "a")
    buf.add("g3", "甲", "b")  # 超过群数上限，丢掉最久没动静的群
    assert buf.recent("g1", 10) == []
    assert buf.recent("nope", 5) == []


def test_state_and_question():
    buf = ContextBuffer()
    buf.add("g", "甲", "你好")
    buf.add("g", "机器人自己", "在呢", is_bot=True)
    state = build_state(["小星", "星星"], buf.recent("g", 8), "乙", "x" * 500, False)
    assert state["bot_name"] == "小星" and state["bot_aliases"] == ["星星"]
    assert state["recent_messages"][1]["speaker"] == "机器人自己"
    assert len(state["current_message"]["text"]) == 200
    questions = build_questions(["小星", "星星"], "问天气的事要回应")
    assert set(questions) == {"to_bot", "to_other", "open", "new_topic"}
    assert "新话题" in questions["new_topic"] and "测试" in questions["new_topic"]
    assert "小星" in questions["to_bot"] and "星星" in questions["to_bot"]
    assert "问天气的事要回应" in questions["to_bot"]
    assert "吃了吗" in questions["open"]
    assert questions_text(questions).startswith("【对机器人说】")
    assert questions_text({"reply": "单个问题"}) == "单个问题"


def test_state_times_reply_to_and_quote():
    now = 10_000.0
    recent = [
        ChatLine("甲", "小星几点了", ts=now - 7200),
        ChatLine("机器人自己", "下午三点。", is_bot=True, ts=now - 7190, reply_to="甲"),
    ]
    plain = build_state(["小星"], recent, "乙", "吃了吗", False)
    assert "time" not in plain["recent_messages"][0]  # 不给 now 就不带时间（被 @ 时的输入不变）
    timed = build_state(["小星"], recent, "乙", "为什么", False, now=now,
                        quote={"speaker": "机器人自己", "text": "下午三点。"})
    assert timed["recent_messages"][0]["time"] == "2小时前"
    assert timed["recent_messages"][1]["replying_to"] == "甲"
    assert "replying_to" not in timed["recent_messages"][0]
    assert timed["current_message"]["quoting"] == {"speaker": "机器人自己", "text": "下午三点。"}
    assert ago_label(5) == "5秒前" and ago_label(125) == "2分钟前" and ago_label(90000) == "1天前"


def test_buffer_save_and_load():
    buf = ContextBuffer(max_per_group=3)
    buf.add("g", "甲", "你好")
    buf.add("g", "机器人自己", "在呢", is_bot=True, reply_to="甲")
    copy = ContextBuffer(max_per_group=3)
    copy.load(buf.to_dict())
    lines = copy.recent("g", 10)
    assert [(l.speaker, l.text, l.is_bot, l.reply_to) for l in lines] == [
        ("甲", "你好", False, ""), ("机器人自己", "在呢", True, "甲")
    ]
    assert [l.ts for l in lines] == [l.ts for l in buf.recent("g", 10)]  # 时间也保留
    copy.load({"g2": [{"text": " "}, "坏数据", {"text": "x", "ts": "abc"}, {"text": "好", "ts": 5}], "g3": None})
    assert [l.text for l in copy.recent("g2", 10)] == ["好"] and copy.recent("g3", 5) == []
    copy.load("坏数据")  # 不出错


def test_find_quoted():
    buf = ContextBuffer()
    buf.add("g", "乙", "我觉得龙井更好喝一点")
    buf.add("g", "机器人自己", "碧螺春吧，清淡些。", is_bot=True)
    buf.add("g", "丙", "好")
    assert buf.find_quoted("g", "碧螺春吧，清淡些").is_bot
    assert buf.find_quoted("g", "我觉得龙井更好").speaker == "乙"  # 引用被截断
    assert buf.find_quoted("g", "好") .speaker == "丙"  # 完全一样才匹配短句
    assert buf.find_quoted("g", "好的呀") is None  # 太短，不做模糊匹配
    assert buf.find_quoted("g", "") is None and buf.find_quoted("none", "x") is None


def test_combine():
    def verdict(to_bot, to_other, open_, chance=0.5, roll=0.9, left=0.0, threshold=0.55):
        return combine(Scores(to_bot, to_other, open_), threshold, chance, roll, left)

    # 对机器人说：回复，冷却也挡不住
    v = verdict(0.9, 0.2, 0.1, left=20)
    assert v.reply and v.stage == "对机器人说" and v.probability == 0.9
    # 「对机器人说」过了阈值，但更像是对别人说的：不回
    v = verdict(0.65, 0.89, 0.15)
    assert not v.reply and v.stage == "对别人说"
    # 开放问题：按概率接话
    assert verdict(0.1, 0.3, 0.9, roll=0.2).reply
    v = verdict(0.1, 0.3, 0.9, roll=0.7)
    assert not v.reply and v.stage == "开放问题" and "这次没接" in v.detail
    assert not verdict(0.1, 0.3, 0.9, chance=0.0, roll=0.0).reply
    # 开放问题在冷却中：不接
    v = verdict(0.1, 0.3, 0.9, roll=0.0, left=12)
    assert not v.reply and v.stage == "冷却中" and "12" in v.detail
    # 像开放问题，但同时很像对别人说的：不算开放问题
    assert verdict(0.03, 0.84, 0.6, roll=0.0).stage == "对别人说"
    assert verdict(0.1, 0.84, 0.9, roll=0.0).stage == "对别人说"
    # 自言自语：不回
    v = verdict(0.12, 0.26, 0.56, roll=0.0)
    assert not v.reply and v.stage == "无需回应"
    assert "对机器人 0.12" in v.detail and "新话题 0.00" in v.detail

    # 新话题：和开放问题一样按概率接话、受冷却限制
    def topic(to_bot, to_other, open_, new, **kw):
        return combine(Scores(to_bot, to_other, open_, new), 0.55, kw.get("chance", 0.5),
                       kw.get("roll", 0.0), kw.get("left", 0.0))

    v = topic(0.04, 0.12, 0.54, 0.87)
    assert v.reply and v.stage == "新话题" and v.probability == 0.87
    assert not topic(0.04, 0.12, 0.54, 0.87, roll=0.9).reply
    assert topic(0.04, 0.12, 0.54, 0.87, left=10).stage == "冷却中"
    # 接着别人的话题、对某个人说：不算
    assert topic(0.05, 0.48, 0.73, 0.04).stage == "对别人说"
    assert topic(0.03, 0.6, 0.2, 0.9).stage == "对别人说"
    # 既是开放问题又是新话题：记作开放问题；对机器人说的优先
    assert topic(0.04, 0.06, 0.94, 0.95).stage == "开放问题"
    assert topic(0.75, 0.13, 0.6, 0.81).stage == "对机器人说"
    # 判定线默认 0.6，可以调
    assert topic(0.07, 0.28, 0.74, 0.69).stage == "开放问题"
    assert topic(0.05, 0.12, 0.62, 0.3).stage == "开放问题"
    assert topic(0.05, 0.12, 0.55, 0.58).stage == "无需回应"
    assert combine(Scores(0.07, 0.28, 0.74, 0.69), 0.5, 1.0, 0.0, open_threshold=0.8).stage == "无需回应"
    assert combine(Scores(0.05, 0.12, 0.55, 0.3), 0.5, 1.0, 0.0, open_threshold=0.5).stage == "开放问题"


def test_clean_name():
    from gate_logic import clean_name

    assert clean_name("‭") == ""
    assert clean_name("‭小红‬") == "小红"
    assert clean_name("小​明 ") == "小明"
    assert clean_name("Tom Lee") == "Tom Lee"
    assert clean_name(None) == ""
    roster = Roster()
    roster.load({"users": {"U": {"name": "‭", "groups": []}}})
    assert roster.name_of("U") == ""


def test_normalize_mentions():
    other = "A" * 32
    bot = "B" * 32
    names = {other: "小明"}
    text, mentioned, at_bot = normalize_mentions(f"<@{other}>，和<@{bot}> 说句话", names, {bot}, "小星")
    assert text == "@小明，和@小星 说句话" and mentioned == ["小明"] and at_bot
    text, mentioned, at_bot = normalize_mentions(
        f'<qqbot-at-user id="{other}" /> 你好 <@!{other}>', names, set(), "小星"
    )
    assert text == "@小明 你好 @小明" and mentioned == ["小明"] and not at_bot
    text, mentioned, _ = normalize_mentions("<@UNKNOWN1> 在吗", {}, set(), "小星")
    assert text == "@某位群友 在吗" and mentioned == ["某位群友"]
    assert normalize_mentions("没有 @ 任何人", {}, set(), "小星") == ("没有 @ 任何人", [], False)


def test_state_mentions_fields_only_when_needed():
    plain = build_state(["小星"], [], "甲", "你好", False)
    assert "mentions" not in plain["current_message"] and "at_bot" not in plain["current_message"]
    full = build_state(["小星"], [], "甲", "@小明 你好", False, mentions=["小明"], at_bot=True)
    assert full["current_message"]["mentions"] == ["小明"] and full["current_message"]["at_bot"]
    at_question = build_at_question(["小星"], "规则X")
    assert "同时也 @ 了别人" in at_question and "规则X" in at_question


def test_parse_llm():
    keys = ["to_bot", "to_other", "open"]
    probs, reason = parse_llm_scores(
        '```json\n{"to_bot": 0.8, "to_other": 0.1, "open": 0.3, "reason": "在叫它"}\n```', keys
    )
    assert probs == {"to_bot": 0.8, "to_other": 0.1, "open": 0.3} and reason == "在叫它"
    # 是/否和字符串也认，超出范围的截断
    probs, _ = parse_llm_scores('{"to_bot": true, "to_other": "0.2", "open": 3}', keys)
    assert probs == {"to_bot": 1.0, "to_other": 0.2, "open": 1.0}
    assert parse_llm_scores('{"reply": "否"}', ["reply"])[0] == {"reply": 0.0}
    assert parse_llm_scores("reply: false", ["reply"])[0] == {"reply": 0.0}
    assert parse_llm_scores("to_bot: 0.7, to_other：0.1, open: 0", keys)[0] == {
        "to_bot": 0.7, "to_other": 0.1, "open": 0.0
    }
    assert parse_llm_scores("", keys)[0] is None
    assert parse_llm_scores("我觉得应该回复", keys)[0] is None
    probs, reason = parse_llm_scores('{"to_bot": 0.5}', keys)  # 漏答的按 0 算
    assert probs == {"to_bot": 0.5, "to_other": 0.0, "open": 0.0} and "to_other、open" in reason
    assert parse_llm_scores('{"answer": 1}', keys)[0] is None  # 一个都没答
    prompt = build_llm_prompt({"to_bot": "陈述A", "open": "陈述B"}, {"x": 1})
    assert "- to_bot：陈述A" in prompt and "to_bot、open" in prompt


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} 个测试全部通过")
