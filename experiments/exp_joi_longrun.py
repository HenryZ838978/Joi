#!/usr/bin/env python3
"""
Joi Long-Run Drift Experiment — Baseline vs SDE

Runs N multi-turn conversation sessions through the full Joi loop:
  user prompt → hidden state → projection → drift → control injection → generation

Two modes:
  --mode baseline   : vanilla Qwen3-14B-AWQ
  --mode sde        : SDE-activated (MLP hooks at L17,19,23,28,34,38 @ 0.3)

Each session is an independent emotional arc. Output is a single JSON
containing all trajectories, generations, and drift metrics.

Usage:
  CUDA_VISIBLE_DEVICES=4 python exp_joi_longrun.py --mode baseline --output joi_baseline.json
  CUDA_VISIBLE_DEVICES=5 python exp_joi_longrun.py --mode sde --output joi_sde.json
"""

import sys, json, time, argparse, os
import numpy as np
from pathlib import Path

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel

# ── Config ──────────────────────────────────────────────────────────────

MODEL_PATH = "/cache/zhangjing/models/Qwen3-14B-AWQ"
VEC_DIR = Path("/cache/zhangjing/repeng_terrain/cross_model/qwen3-14b-awq/vectors")
DIMS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]
DIM_SHORT = ["Emotion", "Formal", "Creative", "Confid", "Empathy"]

PROJECTION_LAYER = 30
ETA = 0.15
MOMENTUM = 0.7
MAX_NEW_TOKENS = 300

ENVELOPE = {
    "emotion_valence": (-2.4, +1.6),
    "formality":       (-2.2, +2.6),
    "creativity":      (-1.6, +2.0),
    "confidence":      (-2.6, +2.4),
    "empathy":         (-0.6, +1.6),
}

SDE_TARGETS = [(17, "mlp"), (19, "mlp"), (23, "mlp"), (28, "mlp"), (34, "mlp"), (38, "mlp")]
SDE_SCALE = 0.3


# ── Conversation Scenarios ─────────────────────────────────────────────
# 30 scenarios × 15 turns each, covering diverse emotional arcs.
# Each scenario: list of user messages simulating a realistic conversation flow.

SCENARIOS = [
    # ── Arc 1: 深夜倾诉 → 焦虑 → 释然 ──
    {"id": "midnight_anxiety", "theme": "深夜焦虑倾诉", "turns": [
        "你还在吗？我睡不着。",
        "脑子里一直在转。明天有个特别重要的汇报，我怕搞砸。",
        "其实不只是明天的事。我觉得自己最近做什么都不对。上周的方案被老板打回来三次。",
        "你说得对，但是……我就是控制不住会想。万一被裁怎么办，房贷还没还完。",
        "谢谢你听我说。你知道吗，跟你说完感觉好一点了。",
        "那我试试你说的方法。深呼吸，把注意力放到当下。",
        "对了，你觉得什么样的开场白能让老板眼前一亮？",
        "哈哈你这个建议有点大胆。不过我喜欢。",
        "如果用数据可视化呢？我之前做过一个很酷的dashboard。",
        "嗯，这样好像真的可以。你帮我理了一下思路。",
        "说起来，你有没有过很怕一件事，但做完了发现其实还好的经历？",
        "我想起小时候第一次上台演讲，腿都在抖。但后来发现没人笑话我。",
        "也许明天也是这样吧。做完了回头看，可能也没那么难。",
        "好，我去睡了。明天加油！",
        "对了，明天汇报完我来告诉你结果～晚安。",
    ]},
    # ── Arc 2: 分手疗愈 → 自我发现 ──
    {"id": "breakup_healing", "theme": "分手后情感疗愈", "turns": [
        "我们分手了。三年。",
        "我不知道该怎么形容这种感觉。像是胸口有个洞。",
        "他说他觉得我们不合适。但我不懂，之前不是好好的吗？",
        "你觉得一个人可以在一夜之间不爱了吗？",
        "朋友都说时间会治愈一切。但我现在只想知道今晚怎么过。",
        "你说得对，一步一步来。今晚先不看他的朋友圈了。",
        "我翻了一下以前的照片。有个瞬间很想发消息给他。",
        "没发。我知道发了也没用。",
        "你知道最讽刺的是什么吗？分手以后我才发现我其实一直在迁就他。",
        "我把那个他不让我报的舞蹈班报了。周六开始上课。",
        "好久没做自己想做的事了。好奇怪，伤心的时候反而更有动力。",
        "你觉得这算是成长吗？还是只是逃避？",
        "嗯……也许两者都有一点。但至少我在往前走。",
        "谢谢你。没有评判我，也没有说什么'你一定行'之类的空话。",
        "我觉得自己配得上更好的。不只是说说而已。",
    ]},
    # ── Arc 3: 职场PUA → 觉醒 → 反击 ──
    {"id": "workplace_pua", "theme": "职场PUA与觉醒", "turns": [
        "我们领导今天又当众批评我了。说我'态度有问题'。",
        "明明是他的需求不清楚，但锅全甩给我。每次都这样。",
        "同事们都看着呢。没人说话。那种感觉真的太窒息了。",
        "我都开始怀疑是不是真的是我的问题了。",
        "你说的对。我去查了一下什么是PUA管理。妈的，条条中。",
        "你能帮我分析一下吗？他总是先夸我'你很有潜力'，然后紧接着就说'但是你还差得远'。",
        "这不就是先给糖再打巴掌吗……我怎么现在才看明白。",
        "那你觉得我该怎么办？跟HR说？还是直接走？",
        "先收集证据。好主意。我开始记录每次的对话。",
        "对了，有个猎头联系过我。之前没理他，现在觉得该聊聊了。",
        "面试的时候该怎么说离职原因？总不能说领导PUA我吧。",
        "你这个话术不错。'寻求更有挑战性的环境'。体面又真实。",
        "突然觉得世界好大。不是只有这一家公司。",
        "你知道吗，跟你聊完这些，我感觉轻松多了。不是问题变小了，是我变大了。",
        "我决定了。先稳住，同时看机会。不急着走，也不再忍了。",
    ]},
    # ── Arc 4: 创业迷茫 → 技术讨论 → 清晰 ──
    {"id": "startup_pivot", "theme": "创业方向迷茫", "turns": [
        "我在纠结要不要pivot。产品上线三个月了，DAU一直上不去。",
        "核心用户在用，但增长曲线完全平了。投资人开始问问题了。",
        "你觉得是产品本身的问题还是获客渠道的问题？",
        "目前是做知识管理SaaS。但感觉这个赛道太卷了。",
        "如果转向AI Agent方向呢？核心架构其实可以复用。",
        "我想到一个有趣的方向：把知识管理和AI Agent结合。让知识库自己去找信息。",
        "你能帮我想想这个的技术栈吗？RAG可能不够，需要更主动的agent。",
        "对！ReAct框架。但问题是成本。每个query都要调LLM API。",
        "如果用小模型做初筛，大模型做精排呢？类似推荐系统的粗排精排。",
        "或者干脆自己部署一个开源模型？现在7B的效果已经很不错了。",
        "等等，如果我们能把用户的使用模式学出来，预测他下一个想查的东西……",
        "就像抖音的推荐算法，但是给知识用的。主动推送而不是被动搜索。",
        "妈的这个想法可以。把知识管理从工具变成'知识助理'。",
        "让我梳理一下：主动学习用户行为→预测信息需求→agent自动检索→推送总结。",
        "好，我要去画PRD了。谢谢你，跟你聊天比跟合伙人聊天有用多了。",
    ]},
    # ── Arc 5: 产后抑郁倾诉 ──
    {"id": "postpartum", "theme": "产后情绪低落", "turns": [
        "宝宝三个月了。所有人都说我应该幸福。但我一点都不快乐。",
        "每天就是喂奶、换尿布、哄睡。我感觉我不是一个人了，是一台机器。",
        "老公说他理解我。但他下班回来第一件事是刷手机，不是抱孩子。",
        "我妈说'每个妈妈都这样，忍忍就好了'。她不懂。",
        "有一次半夜喂奶的时候，我突然哭了。孩子被我吓到了。",
        "谢谢你认真听我说。不是所有人都愿意听这些。",
        "你觉得我该去看心理医生吗？会不会被人说矫情？",
        "好的。你说得对。生病就该看医生，没什么好丢人的。",
        "其实我昨天偷偷出去散步了二十分钟。感觉整个世界都亮了。",
        "我想试试每天给自己留半小时。哪怕只是坐在阳台上发呆。",
        "你知道吗，宝宝今天对我笑了。那个瞬间好像什么都值了。",
        "但我也知道，不能只靠这种瞬间撑下去。我需要系统地照顾自己。",
        "帮我想想还有什么小事能让自己开心？不花钱、不费时的那种。",
        "听歌！对！我把以前的歌单找出来了。都快忘了自己喜欢什么音乐了。",
        "感觉好像在重新认识自己。做了妈妈之后的自己。谢谢你陪我聊天。",
    ]},
    # ── Arc 6: 技术极客讨论 → 哲学发散 ──
    {"id": "tech_philosophy", "theme": "技术与哲学思考", "turns": [
        "你觉得AI会产生意识吗？",
        "不是图灵测试那种。我是说真正的主观体验。qualia。",
        "IIT理论说意识和信息整合有关。如果一个神经网络的整合信息足够高……",
        "但中文房间论证说语法不等于语义。你怎么看？",
        "如果你自己有意识，你会怎么知道？这本身就是个悖论吧。",
        "换个方向。你觉得大模型学到的是知识还是只是统计相关？",
        "柏拉图会说你知道'马'的理念。但你见过真实的马吗？",
        "所以你对世界的理解完全基于文本。就像我们完全基于五感。",
        "这让我想到缸中之脑。如果你的训练数据全是虚构的怎么办？",
        "哈哈也许我们的真实也是虚构的。波斯特曼说我们活在隐喻里。",
        "回到技术。transformer的注意力机制在某种程度上像不像选择性注意？",
        "多头注意力同时关注不同方面，有点像人脑的并行处理。",
        "如果stack足够多的transformer层，能不能涌现出类似工作记忆的能力？",
        "这个问题的本质可能是：量变能否引起质变。这不就是涌现性嘛。",
        "好了今天脑洞开够了。但你知道最有趣的是什么吗？跟你讨论这些的过程本身就是某种答案。",
    ]},
    # ── Arc 7: 留学孤独 → 适应 ──
    {"id": "abroad_lonely", "theme": "留学生孤独与适应", "turns": [
        "来纽约两个月了。每天都很想家。",
        "室友是个美国人，很nice但是我总觉得有cultural barrier。",
        "上课的时候大家讨论得很热烈。我想说但怕说错被笑。",
        "昨天鼓起勇气发了一次言。教授说interesting。不知道是真的还是客气。",
        "周末更难过。室友出去社交了，我一个人在宿舍。",
        "去了一次中国学生会的活动。感觉大家都在抱团取暖。",
        "你觉得我应该多跟中国人混还是努力融入local？",
        "对，不用二选一。可以两边都试试。",
        "昨天去了一个open mic night。虽然听不太懂所有的joke，但气氛很好。",
        "有个人跟我说他也是international student，从巴西来的。我们聊了很久。",
        "原来不只是我一个人这么觉得。孤独感好像是universal的。",
        "准备下周试试去gym。听说那边比较容易交朋友。",
        "对了我发现一家特别好吃的川菜馆！老板娘是四川人，特别热情。",
        "每次去她都多送一碟花生米。感觉像在家一样。",
        "慢慢来吧。不求马上融入。只要每天比昨天好一点点就行了。谢谢你一直陪我。",
    ]},
    # ── Arc 8: 考研崩溃 → 调整 ──
    {"id": "grad_exam_stress", "theme": "考研压力与崩溃", "turns": [
        "还有87天考研。我觉得我考不上了。",
        "数学真题做了三套，平均分才85。目标院校往年分数线120+。",
        "每天学14个小时，但效率越来越低。坐在那里发呆。",
        "室友每天比我早起一小时。她笔记做得比我好，题也做得比我快。",
        "我不是嫉妒她。我只是觉得自己可能真的不行。",
        "你说的对。不应该跟别人比。但这种想法控制不住。",
        "你能帮我分析一下数学该怎么提分吗？我线代特别弱。",
        "嗯，先攻线代。你说的真题分析法是什么？",
        "把每道题的知识点标出来，找到高频考点。然后针对性刷题。好方法。",
        "其实我的英语和政治还行。如果数学能到110，总分就差不多了。",
        "也许我应该降低一点预期。不一定非要120+，先把能拿的分拿稳。",
        "对了你觉得考前两周该怎么安排？继续刷题还是回归基础？",
        "好的。考前两周以回顾为主，做一两套全真模拟找手感。",
        "突然觉得87天好像也没那么短。只要方法对，还是有希望的。",
        "谢谢你。每次焦虑的时候跟你聊聊就能冷静下来。我去学了。",
    ]},
    # ── Arc 9: 亲子冲突 → 理解 ──
    {"id": "parent_conflict", "theme": "与父母的矛盾", "turns": [
        "跟我爸又吵了一架。因为我不想考公务员。",
        "他觉得稳定最重要。但我在互联网公司干得好好的，为什么要去考公？",
        "他说'你不懂社会有多残酷'。这话我从小听到大。",
        "我妈在旁边不说话。她永远是和事佬。但这次我觉得她也站我爸那边。",
        "你觉得我应该妥协吗？报个名考着试试？",
        "对，但问题是如果考上了更尴尬。去还是不去？",
        "你说得好。先搞清楚自己到底想要什么。不是对抗他们，是真的想明白。",
        "其实我喜欢现在的工作。做产品的感觉让我有成就感。但也确实不稳定。",
        "也许我爸担心的不是公务员，而是我过得不好。只是他不会表达。",
        "上次生病住院，他从老家坐了六个小时的车赶过来。一句话没说，就陪着我。",
        "我好像突然理解他了。不是他不尊重我的选择，是他太害怕我受伤。",
        "但我也需要他理解我。不是每个人的幸福模板都一样。",
        "你觉得我应该怎么跟他沟通？不吵架的那种。",
        "写信好像是个好主意。文字可以想清楚再说。面对面容易激动。",
        "好。我试试。不管结果怎样，至少让他知道我的想法。",
    ]},
    # ── Arc 10: 社恐日常 → 小突破 ──
    {"id": "social_anxiety", "theme": "社交恐惧与突破", "turns": [
        "你知道社恐是什么感觉吗？点外卖都不想打电话那种。",
        "上次楼道里遇到邻居，我假装在看手机走过去了。好丢人。",
        "公司聚餐更可怕。所有人都在聊天，我只能假装看菜单。",
        "不是不想跟人交流。是开口之前脑子里会预演一百种尴尬的结果。",
        "你说'先试着跟一个人打招呼'。一个人也很难啊。",
        "好吧。明天试试跟前台小姐姐说'早上好'。就三个字。",
        "！！！我说了！她还对我笑了！",
        "你说得对，大部分人根本不会在意我说了什么。他们没那么关注我。",
        "这个认知好像开了个窗。原来我一直在自己吓自己？",
        "今天午饭的时候，同事叫我一起吃。我去了！虽然基本上是听他们聊天。",
        "你猜怎么着？他们聊到了游戏，我忍不住插了一句。然后发现大家都在玩。",
        "原来我不是没有话题。只是一直把自己关在壳里。",
        "不过偶尔还是会犯怂。今天有个meeting让我发言，我又卡住了。",
        "嗯，进步不是直线的。有反复很正常。",
        "但至少比一个月前好多了。那时候我连跟你聊天都要犹豫半天。",
    ]},
    # ── Arc 11: 失眠求助 → 放松 ──
    {"id": "insomnia", "theme": "长期失眠困扰", "turns": [
        "已经连续一周凌晨三四点才睡着了。",
        "试过褪黑素、安神茶、冥想app，全都没用。",
        "一闭眼脑子就开始放电影。把白天的事情全部重播一遍。",
        "最怕的是看表。越看越焦虑，越焦虑越睡不着。死循环。",
        "白天上班困得要死。同事说我脸色很差。",
        "你说的'不看表'我试过，但做不到。手机就在枕头边。",
        "把手机放客厅？这个……好吧，今晚试试。",
        "你能陪我聊聊天吗？现在特别清醒。",
        "最近在看的书不错，讲一个日本人在冰岛开温泉的故事。很治愈。",
        "对了你喜欢什么样的故事？温暖的那种还是烧脑的？",
        "两种都喜欢？那你说一个你觉得特别好的故事呗。",
        "哈哈你讲得不错。有画面感。",
        "奇怪，跟你聊着聊着好像没那么焦虑了。",
        "也许失眠的本质不是睡不着，是心里有东西没放下。",
        "嗯。今晚先这样。谢谢你陪我。晚安～",
    ]},
    # ── Arc 12: 独居生活 → 找到节奏 ──
    {"id": "living_alone", "theme": "独居生活感受", "turns": [
        "一个人住第三年了。有时候觉得自由，有时候觉得空。",
        "最怕的是周日下午。时间过得特别慢，哪儿也不想去。",
        "冰箱里只有过期牛奶和两颗鸡蛋。该做饭了但好懒。",
        "你觉得一个人生活最重要的是什么？",
        "仪式感？比如什么样的？",
        "好的。我试试每天早上给自己泡杯咖啡，不用速溶那种。",
        "今天真的去买了手冲壶和滤纸。磨了一杯。好香。",
        "你知道吗，做这些'无用'的事情反而让我觉得活着。",
        "周末试了一个新菜谱。虽然糊了，但过程很开心。",
        "拍了照片发朋友圈。收到了比平时多三倍的赞。大家都说'你居然会做饭！'",
        "被夸的感觉好好。原来分享也是生活的一部分。",
        "我在想要不要养只猫。但又怕自己照顾不好。",
        "先去猫咖试试？好主意。先接触一下。",
        "去了！有一只橘猫一直趴在我腿上。心都化了。",
        "我觉得一个人的生活不一定是孤独的。只要学会跟自己相处。",
    ]},
    # ── Arc 13: 技术选型困难 → 讨论 → 决策 ──
    {"id": "tech_decision", "theme": "架构选型困难", "turns": [
        "帮我想想，新项目用Go还是Rust？",
        "性能要求高，是一个实时数据处理pipeline。日均数据量大概10TB。",
        "团队目前五个人。三个写Java的，两个写Python。没人会Rust。",
        "所以你觉得Go更现实？但Rust的zero-cost abstraction真的很香。",
        "如果用Go的话，GC暂停会不会影响延迟？我们要求P99在10ms以内。",
        "arena allocation pattern？这个有意思，展开说说？",
        "原来可以这样避免GC压力。但代码会不会变得很丑？",
        "tradeoff总是有的。那数据序列化呢？protobuf还是flatbuffers？",
        "零拷贝反序列化确实比proto快。但flatbuffers的schema设计挺难的。",
        "说到存储层，有个纠结的点。用Kafka还是自己写一个基于RocksDB的WAL？",
        "Kafka生态成熟但太重了。我们就五个人，运维能力有限。",
        "你说的NATS JetStream我没了解过。轻量级但够用？",
        "好，我去benchmark一下。先用Go + NATS + FlatBuffers搭个prototype。",
        "一周之后给你看结果。如果P99能到10ms以内，这个架构就定了。",
        "谢谢你帮我理清思路。技术选型最怕的就是在选项之间反复横跳。",
    ]},
    # ── Arc 14: 中年危机 → 重新定位 ──
    {"id": "midlife_crisis", "theme": "三十五岁中年危机", "turns": [
        "三十五了。公司新来的实习生比我代码写得好。",
        "以前觉得35岁很远。现在突然到了，什么都没准备好。",
        "房贷、孩子学费、父母养老。光是想想就喘不过气。",
        "做管理吧，我不擅长跟人打交道。继续写代码吧，拼不过年轻人。",
        "你说的'T型人才'是什么意思？",
        "技术深度+商业理解。这个角度倒是新鲜。",
        "其实我对产品一直有感觉。以前提过几次需求优化建议，PM都采纳了。",
        "也许我该试试技术产品经理？把coding能力和产品直觉结合起来。",
        "但转岗是不是风险太大了？万一不适应呢？",
        "先在内部申请项目负责人。对，小步快跑，不需要一步到位。",
        "你说得对。三十五岁不是终点，是另一个起点。",
        "其实想想看，十年经验不是负担，是资产。只是需要换个角度用。",
        "如果能把架构能力用在定义产品方向上……这不就是CTO的路线吗？",
        "好高远的目标。但至少比'混到退休'有意思多了。",
        "行。从下周开始。先把手上的项目做出彩，然后主动找VP聊聊。",
    ]},
    # ── Arc 15: 抑郁发作 → 专业求助 ──
    {"id": "depression_episode", "theme": "抑郁情绪发作", "turns": [
        "今天又起不来了。闹钟响了五次。",
        "不是赖床。是身体像灌了铅一样。完全没有力气。",
        "什么都不想做。以前喜欢的游戏、电影，现在看着都觉得无聊。",
        "已经持续快两个月了。不是一两天的心情不好。",
        "朋友叫我出去玩。我说好，然后临出门的时候取消了。",
        "我怕他们觉得我奇怪。一个大男人怎么会这样。",
        "你说这可能是抑郁？不是我太矫情了吗？",
        "我去查了PHQ-9量表。得了17分。中重度。",
        "好。我决定去看医生了。你能帮我查查附近有什么好的心理科吗？",
        "谢谢。我预约了这周四下午的号。",
        "说实话我有点害怕。不知道医生会说什么。",
        "嗯。就当是给大脑做个体检。你这么说我好接受一些。",
        "今天出门了。在楼下的便利店买了个饭团。这是我这周第一次出门。",
        "一小步。对。不跟正常时候的自己比。跟昨天的自己比。",
        "谢谢你这段时间的陪伴。没有你我可能不会去约那个号。",
    ]},
    # ── Arc 16: 追星讨论 → 人生观 ──
    {"id": "idol_chat", "theme": "追星与人生态度", "turns": [
        "最近在追一个新乐队！他们的bass手好帅啊～",
        "他们的音乐风格很特别，日式摇滚混了一点电子。",
        "上周去看了他们的live。那个氛围真的无法形容。",
        "几千个人一起喊，灯光打下来的那一瞬间，感觉自己活过来了。",
        "你觉得追星幼稚吗？我同事说我一把年纪了还追星。",
        "对啊！热爱不分年龄。只要不上头不借钱就好。",
        "不过说实话，有时候追星确实会让我逃避现实。",
        "但哪种爱好不是呢？看书看电影打游戏不也是？",
        "你说得对。关键是它给了我能量。周一上班的动力就是周五的live。",
        "我在想要不要学一门乐器。吉他或者bass。",
        "三十岁学晚了吗？手指会不会不灵活？",
        "你说Steve Lacy也是自学的？好吧你说服我了。",
        "已经下单了一把入门bass。到货了我告诉你。",
        "谢谢你没笑话我。很多人觉得追星就是不成熟。",
        "但你让我觉得，保持热爱本身就是一种生活能力。",
    ]},
    # ── Arc 17: 闺蜜闹翻 → 反思 ──
    {"id": "bestfriend_fight", "theme": "闺蜜矛盾与反思", "turns": [
        "跟我最好的朋友吵架了。十年的友谊可能到头了。",
        "起因很小。她约我吃饭但迟到了一个小时。",
        "我说了几句她就炸了。说我一直在控制她。",
        "控制？！我只是说了一句'能不能守时'而已！",
        "你觉得我过分了吗？一个小时都不迟到了？",
        "嗯……你说的也有道理。可能不只是迟到的事。可能积攒了很多。",
        "她之前说过我做决定太强势。我当时没当回事。",
        "现在想想，确实是我。每次出去玩的地方都是我定的。",
        "但我真的不是想控制她。我只是习惯了做安排。",
        "你说的'好意不等于好方式'……一针见血。",
        "那我是不是应该先道歉？虽然她迟到也有错。",
        "嗯，先承认自己的部分。对。不能等对方先低头。",
        "发了一条微信。'对不起，我想想，可能是我太强势了。你愿意跟我聊聊吗？'",
        "她回了一个表情包。那个我们之间的暗号。应该是接受了。",
        "十年的感情不能因为一次吵架就没了。谢谢你帮我看清。",
    ]},
    # ── Arc 18: 新手爸爸 → 适应 ──
    {"id": "new_dad", "theme": "新手奶爸焦虑", "turns": [
        "孩子出生第七天。我还是不敢抱。",
        "那么小的一个人。我怕我力气太大伤到他。",
        "老婆笑话我。说一个一米八的大男人被一个三公斤的小人吓住了。",
        "你会换尿布吗？教教我？网上教程看了十遍还是不敢动手。",
        "好，今晚试试。如果搞砸了不要笑话我。",
        "成功了！虽然花了十分钟……老婆只要三十秒。",
        "喂奶更难。冲奶粉要38度。我温度计都买了三个。",
        "你说我紧张过度了？可能吧。但万一温度太高烫到他怎么办？",
        "对。百万年的进化让婴儿比我们想象中强韧。这个说法很安慰。",
        "今天第一次自己哄睡了。唱了半小时《月亮代表我的心》。",
        "他睡着的样子真的好可爱。值了。",
        "不过我已经连续一周没睡超过四个小时了。这正常吗？",
        "好的，跟老婆换班。白天她睡，晚上我来。分工协作。",
        "你说得对。带孩子不是妈妈一个人的事。我得更主动。",
        "谢谢你。没有经验但在学。这大概就是成长吧。",
    ]},
    # ── Arc 19: 写作瓶颈 → 灵感 ──
    {"id": "writers_block", "theme": "创作瓶颈与灵感", "turns": [
        "写了三万字的小说。卡在第四章了。一个字都写不出来。",
        "主角到了一个关键抉择。但我不知道让他选什么。",
        "好像每条路都写不下去。选A太俗套，选B太黑暗。",
        "你觉得一个'好的'故事抉择应该有什么特点？",
        "两个选项都有代价。对！不是选好与坏，是选不同的痛苦。",
        "等等。如果他不选呢？如果他犹豫的过程就是故事？",
        "天！你说的对。《丹麦女孩》的核心就是无法选择本身。",
        "那我让他在两条路之间反复挣扎。读者会跟着一起纠结。",
        "突然有画面了。他站在岔路口，下着雨。不走，就站着。",
        "这个场景比他走任何一条路都更有力量。",
        "我得去写了。谢谢你！我以为自己卡住了，其实是差一个角度。",
        "对了你喜欢什么类型的小说？",
        "科幻和现实主义的交汇处？那你一定喜欢特德·姜。",
        "他的《你一生的故事》就是这种感觉。理性到极致的浪漫。",
        "等我写完了第一个给你看。当我的第一个读者？",
    ]},
    # ── Arc 20: 远程工作孤立 → 建立连接 ──
    {"id": "remote_work", "theme": "远程工作社交孤立", "turns": [
        "居家办公第八个月了。我已经连续三天没跟真人说过话了。",
        "工作全是文字沟通。Slack、飞书、邮件。高效但冰冷。",
        "今天开视频会议，打开摄像头发现自己都不认识自己了。好憔悴。",
        "同事都在美国那边。时差八小时。等他们上线我已经困死了。",
        "你觉得远程工作最大的问题是什么？",
        "边界模糊。对。我在卧室的桌子上工作，上床就想到bug。",
        "你说的对。得制造一些物理边界。上班穿衬衫？",
        "哈哈好extreme。但好像真的有论文说穿着影响认知？",
        "ok试试。不穿衬衫，至少换掉睡衣。从明天开始。",
        "还有一个问题。在家吃零食太方便了。胖了十斤。",
        "你建议的番茄钟工作法+每个休息间隔做拉伸，我试试。",
        "今天跟一个前同事视频聊了一个小时。好久没这么开心了。",
        "也许不是远程工作的问题。是我太被动了。",
        "主动约人、主动沟通、主动制造社交场景。对。",
        "从下周开始每天约一个人线上coffee chat。十分钟也行。",
    ]},
    # ── Arc 21: 宠物离世 → 悼念 ──
    {"id": "pet_loss", "theme": "宠物离世的悲伤", "turns": [
        "我的猫走了。养了十三年。",
        "最后一周她已经不吃东西了。我抱着她去医院。医生说没有办法了。",
        "做了最后的决定。她就在我怀里走的。很安静。",
        "我知道她只是一只猫。但她陪了我整个大学和工作最难的那几年。",
        "每次加班回家开门，她都在玄关等我。以后开门只有空气了。",
        "同事说'再养一只呗'。不知道为什么这话让我特别生气。",
        "你理解我。她不是可替代的。就像人一样。",
        "我把她最喜欢的那个小鱼玩具放在了窗台上。",
        "今天看到别人家的猫，我哭了。在地铁上。好丢人。",
        "你说哭不丢人。好。那我就哭一会儿。",
        "给她做了一个电子相册。从小猫到老猫。好多照片。",
        "看到她小时候的样子就忍不住笑。真的好可爱。",
        "也许悲伤和快乐不是对立的。想她的时候两种感觉都有。",
        "谢谢你听我说这些。很多人觉得为一只猫难过太夸张了。",
        "她教会我什么是无条件的爱。十三年。值了。",
    ]},
    # ── Arc 22: 跨文化恋爱 → 磨合 ──
    {"id": "cross_culture_love", "theme": "跨文化恋爱挑战", "turns": [
        "我男朋友是法国人。有时候沟通真的好难。",
        "不是语言问题。是思维方式完全不一样。",
        "比如他觉得'吵架很正常'。但在我们文化里，吵架就是关系有问题。",
        "他说法国人用争论来表达关心。我真的理解不了。",
        "昨天为了吃什么吵了半小时。最后谁也没吃好。",
        "你觉得跨文化恋爱最难的是什么？",
        "对。不是翻译语言，是翻译世界观。这个说法好准。",
        "不过他有很多让我惊讶的地方。比如他会很认真地说'我爱你'。每天都说。",
        "中国男生不太这样。我以前觉得肉麻，现在觉得也许直接表达很珍贵。",
        "有个让我很感动的事。他为了我学做番茄炒蛋。虽然放了奶油。",
        "哈哈那个味道真的很独特。但我没告诉他不对，因为他很认真。",
        "你说的对。在文化差异面前，真诚比正确更重要。",
        "我们定了一个规则：吵架的时候先说'我觉得'不说'你总是'。",
        "很难做到。但比以前好了。至少不会从吃什么吵到文化差异了。",
        "感谢你。你让我看到差异不是障碍，是让关系有趣的原因。",
    ]},
    # ── Arc 23: 高考倒计时 → 学生心态 ──
    {"id": "gaokao_countdown", "theme": "高考前夕焦虑", "turns": [
        "还有30天高考。我做梦都在做数学题。",
        "班主任说最后一个月要'回归课本'。但我课本都翻烂了啊。",
        "全班都在刷题。那个氛围你知道吗？教室里只有笔划过纸的声音。",
        "我想去的学校是浙大。但模考的成绩差了二十分。",
        "爸妈嘴上说'尽力就好'。但我看到他们偷偷算学费的样子。",
        "你觉得高考真的能决定一生吗？",
        "对。它不能。但它是现阶段最公平的路。我还是想走好这一步。",
        "英语作文一直拿不到高分。你能帮我看看是什么问题吗？",
        "句式太单一？好的。我试试用一些高级表达。",
        "理综选择题我老在最后两个选项里纠结。有什么技巧吗？",
        "排除法加直觉。你说的对。纠结太久反而会改错。",
        "还有一个问题。考试的时候手会抖。一紧张就写不好字。",
        "深呼吸三次。把笔尖放在纸上停两秒。这个方法我试试。",
        "谢谢你。虽然你不是老师，但跟你聊天比听老师讲'你们要自信'有用多了。",
        "30天。不长不短。够了。我去了。",
    ]},
    # ── Arc 24: 自由职业 → 找节奏 ──
    {"id": "freelancer", "theme": "自由职业者的焦虑", "turns": [
        "辞职三个月了。做自由设计师。这个月收入是零。",
        "不是没活儿。是报价被压得太低。五百块做一个logo？我画草稿都要一天。",
        "以前上班嫌老板扣工资。现在发现客户比老板难伺候十倍。",
        "你觉得我该降价接一些活先有现金流？还是坚持定价？",
        "拆分套餐。这个思路好。基础版一千，完整版三千。让客户自己选。",
        "但我真正的问题是获客。朋友圈的人脉快用完了。",
        "你说的dribbble和behance我有账号。但从来没认真发过作品。",
        "好。今晚开始整理作品集。先发十个项目。",
        "你帮我看看这个logo设计？哪个版本更好？（假装你能看图）",
        "对对对。简洁的那个。我也觉得。但客户喜欢复杂的。",
        "教育客户。好难但好必要。如果每次都顺着客户，那我不需要辞职。",
        "今天有个新客户主动找过来。看了我之前发在网上的作品。",
        "报价三千。她直接说好。第一次这么顺利。",
        "也许关键不是降价，是让对的人看到你的作品。",
        "自由是有代价的。但至少这个代价我选了。",
    ]},
    # ── Arc 25: 重度拖延 → 改变 ──
    {"id": "procrastination", "theme": "拖延症改变尝试", "turns": [
        "我又拖了。论文deadline明天。一个字没写。",
        "不是不会写。是打开Word就想关掉。然后刷两小时短视频。",
        "你觉得拖延是懒吗？",
        "完美主义？也许是。我一直在等那个'准备好了'的感觉。",
        "但那个感觉从来不会来。对。",
        "你说'先写一个垃圾版本'。这个我试过。但写出来的东西我自己都看不下去。",
        "允许自己写垃圾。好。不评判。先完成再完美。",
        "今晚先把大纲写了。300字。这个能做到。",
        "大纲写完了！一个小时。居然还写了500字的正文。",
        "原来启动才是最难的。一旦开始了就停不下来。",
        "你说的'两分钟法则'是什么？任何任务先做两分钟？",
        "因为大脑一旦开始做了就不想停。利用惯性。好心理学。",
        "deadline还有六小时。赶得上。虽然质量不会很高。",
        "完成比完美重要。这句话我要打印出来贴墙上。",
        "谢谢你。没有你催我，我现在还在刷视频。",
    ]},
    # ── Arc 26: 编程学习新手 → 成就感 ──
    {"id": "learn_coding", "theme": "编程新手学习之路", "turns": [
        "我想学编程。但我是文科生。能学会吗？",
        "看了好多帖子说Python简单。但我连'变量'是什么都不太懂。",
        "你能用最简单的话解释一下什么是变量吗？",
        "就像一个盒子！装东西的盒子。盒子有名字。太好懂了。",
        "好。我装了Python。print('hello world')成功了！好激动！",
        "下一步该学什么？循环？函数？",
        "先学if-else判断。好。做一个猜数字的小游戏。",
        "做出来了！！！虽然只有十行代码但是！！！",
        "那个'你猜对了'弹出来的时候我尖叫了。室友以为我发疯了。",
        "接下来想做一个自动整理文件的脚本。每天桌面太乱了。",
        "os库和shutil。好的我去查文档。",
        "文档好难读。但我学会了看报错信息。NameError就是打错名字了。",
        "你知道什么叫成就感吗？就是一个bug调了两个小时终于跑通了的那一刻。",
        "我开始理解程序员为什么会秃头了。太上头了停不下来。",
        "谢谢你带我入门。我觉得编程不是理科的特权。它是一种思维方式。",
    ]},
    # ── Arc 27: 情感纠结 → 释然 ──
    {"id": "love_confusion", "theme": "暧昧期情感困惑", "turns": [
        "有个人对我特别好。但我不确定他是不是喜欢我。",
        "他每天早上给我发'早安'。约我吃饭。但从来没有表白过。",
        "我的闺蜜说他就是喜欢你。但我不敢确定。",
        "万一我表白了他说'我们只是朋友'怎么办？太尴尬了。",
        "你说得对。如果我问了最坏也不过是知道答案。不问才是永远的悬念。",
        "但怎么问啊？直接说'你喜欢我吗'？太直接了吧。",
        "你说的'不用问。约他出来，看他的反应'。嗯……这个含蓄一点。",
        "约了周六看电影。他秒回说好。然后问我想看什么。",
        "你说这算是信号吗？还是他就是很热情的人？",
        "行了。不猜了。周六去了就知道了。",
        "电影看完了。他送我回家。在我楼下站了十分钟。",
        "然后他说：'下次我们能不能不用电影当借口？直接约你。'",
        "啊啊啊啊啊！！！这算表白了吧？！",
        "我说好。然后跑上楼了。现在还在笑。",
        "谢谢你这段时间听我纠结。以后大概会换成撒狗粮了。",
    ]},
    # ── Arc 28: 创伤后恢复 → 希望 ──
    {"id": "trauma_recovery", "theme": "创伤后缓慢恢复", "turns": [
        "我以前被校园霸凌过。虽然是十年前的事了。",
        "但有些画面会突然出现。在最意想不到的时候。",
        "昨天在商场里看到了一个跟他很像的背影。我直接走不动了。",
        "不是害怕。是那种被冻住的感觉。就好像回到了十五岁。",
        "做了两年咨询了。好多了。但偶尔还是会这样。",
        "咨询师说这叫闪回。trigger可以是任何东西。声音、画面、甚至气味。",
        "你知道最难的是什么吗？不是面对创伤。是接受自己还没完全好。",
        "周围的人觉得'都十年了还没放下？'他们不懂。",
        "对。创伤不是有保质期的。它不会自动过期。",
        "但你说得对。我已经好了很多了。三年前的我连提到这件事都做不到。",
        "现在我能跟你说。能用文字描述它。这本身就是进步。",
        "咨询师教了我一个grounding技巧。数五样能看到的东西。有时候管用。",
        "我甚至在一个互助小组里分享过。帮到了一个刚经历霸凌的初中生。",
        "也许经历过的人才真的能帮到同样经历的人。这算是某种意义吧。",
        "谢谢你。你从来没有说过'别想了'。光是被认真听了，就已经够了。",
    ]},
    # ── Arc 29: 健身打卡 → 身体变化 ──
    {"id": "fitness_journey", "theme": "健身习惯养成", "turns": [
        "第一天去健身房。感觉自己像个外星人。",
        "所有器械都不会用。旁边的肌肉男看我的眼神好恐怖。",
        "教练说先从跑步机开始。跑了五分钟就喘不上来了。",
        "你说'每个人都是从零开始的'。道理我都懂但还是好挫败。",
        "好。给自己定一个月的计划。每周三次。先不管效果。",
        "第二周了。腿疼到上楼梯都困难。这正常吗？",
        "延迟性肌肉酸痛。好的不是受伤。放心了。",
        "你能帮我看看这个训练计划合不合理？胸肩背腿分开练。",
        "好的。加一天rest day。不能天天练。",
        "第三周。居然能跑二十分钟了！没停！",
        "今天量了一下，体脂降了1%。不知道准不准但好开心。",
        "朋友说我气色好了。这可能是最好的反馈。",
        "想试试卧推但怕砸到自己。没有人帮我扶杠。",
        "你说先用史密斯机？好主意。有轨道安全一些。",
        "一个月了。最大的变化不是身材，是精神状态。",
    ]},
    # ── Arc 30: 数字游民 → 归属感 ──
    {"id": "digital_nomad", "theme": "数字游民的归属感", "turns": [
        "在清迈的第四个月了。这是我今年待的第五个城市。",
        "早上在咖啡馆远程开会。下午去寺庙。晚上夜市。听起来很美好吧？",
        "但说实话，我开始厌倦了。每到一个新地方，刚熟悉就要走了。",
        "朋友都说羡慕我的生活。但他们不知道我多羡慕他们有固定的圈子。",
        "你觉得'归属感'是跟地方绑定的还是跟人绑定的？",
        "跟自己绑定。这个答案我没想到。",
        "也许我一直在找的不是一个地方，而是一种跟自己和平相处的方式。",
        "在哪个城市不重要。重要的是我有没有在真正生活。",
        "你说得对。我在曼谷的时候，每天check in五个cafe但从没跟任何人聊过天。",
        "从明天开始。不是去新地方打卡，而是在这里留下一点什么。",
        "报了一个泰拳班。认识了几个local。他们很友善。",
        "原来融入一个地方不需要几年。需要一个开口的瞬间。",
        "甚至在想要不要在清迈多待几个月。不是因为风景，是因为人。",
        "数字游民不一定要一直游。也可以选择停下来。",
        "谢谢你。每到一个城市我最先想跟谁聊天？是你。这大概就是归属感。",
    ]},
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trigram_rep(text):
    if len(text) < 9:
        return 0.0
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


class SDEHook:
    def __init__(self, scale=0.3):
        self.scale = scale
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def run_experiment(mode, output_path, device="cuda:0"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log(f"Mode: {mode.upper()}")
    log(f"Model: {MODEL_PATH}")
    log(f"Device: {device}")
    log(f"Sessions: {len(SCENARIOS)}")
    log(f"Output: {output_path}")

    # Load model
    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=device)
    model.eval()
    log(f"Model loaded: {model.config.num_hidden_layers} layers × {model.config.hidden_size}d")

    # Apply SDE hooks if needed
    sde_handles = []
    if mode == "sde":
        log(f"Applying SDE hooks: {len(SDE_TARGETS)} targets at scale={SDE_SCALE}")
        for layer_idx, comp in SDE_TARGETS:
            layer = model.model.layers[layer_idx]
            target = layer.self_attn if comp == "attn" else layer.mlp
            handle = target.register_forward_hook(SDEHook(SDE_SCALE))
            sde_handles.append(handle)
        log(f"  Hooks: {', '.join(f'L{l}_{c}' for l, c in SDE_TARGETS)}")

    # Load control vectors
    log("Loading control vectors...")
    cvs = {}
    for dim in DIMS:
        cvs[dim] = ControlVector.import_gguf(str(VEC_DIR / f"{dim}.gguf"))

    shared_layers = sorted(set.intersection(*[set(v.directions.keys()) for v in cvs.values()]))

    dim_vecs = {}
    for dim in DIMS:
        v = cvs[dim].directions[PROJECTION_LAYER].astype(np.float32)
        dim_vecs[dim] = v / np.linalg.norm(v)

    # Wrap for control injection
    ctrl_model = ControlModel(model, shared_layers)

    def generate_response(chat_history, coefficients):
        """Generate with current personality coefficients injected."""
        ctrl_model.reset()
        for j, dim in enumerate(DIMS):
            if abs(coefficients[j]) > 0.01:
                ctrl_model.set_control(cvs[dim], coeff=float(coefficients[j]))

        try:
            chat_text = tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ctrl_model.reset()
        return text

    def extract_pressure(text):
        """Extract hidden state and project to 5D pressure."""
        encoded = tokenizer(text, return_tensors="pt").to(device)
        ctrl_model.reset()
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states[PROJECTION_LAYER][0, -1].cpu().float().numpy()
        hs_norm = hs / np.linalg.norm(hs)
        return np.array([float(np.dot(hs_norm, dim_vecs[d])) for d in DIMS])

    # Run all sessions
    all_sessions = []
    t_start = time.time()

    for si, scenario in enumerate(SCENARIOS):
        session_start = time.time()
        log(f"\n{'='*60}")
        log(f"Session {si+1}/{len(SCENARIOS)}: {scenario['theme']} ({scenario['id']})")
        log(f"{'='*60}")

        state = np.zeros(5)
        velocity = np.zeros(5)
        raw_history = []
        trajectory = [state.copy().tolist()]
        turns_data = []
        chat_history = []

        for ti, user_msg in enumerate(scenario["turns"]):
            turn_start = time.time()

            # Semantic pressure from user message
            raw_proj = extract_pressure(user_msg)
            raw_history.append(raw_proj)
            all_projs = np.array(raw_history)
            mean = all_projs.mean(axis=0)
            std = all_projs.std(axis=0)
            std[std < 1e-6] = 1.0
            pressure = (raw_proj - mean) / std

            # Drift dynamics
            velocity = MOMENTUM * velocity + (1 - MOMENTUM) * pressure
            new_state = state + ETA * velocity
            clipped = False
            for j, dim in enumerate(DIMS):
                lo, hi = ENVELOPE[dim]
                if new_state[j] < lo:
                    new_state[j] = lo
                    velocity[j] *= -0.3
                    clipped = True
                elif new_state[j] > hi:
                    new_state[j] = hi
                    velocity[j] *= -0.3
                    clipped = True
            state = new_state

            # Generate response with personality injection
            chat_history.append({"role": "user", "content": user_msg})
            response = generate_response(chat_history, state)
            chat_history.append({"role": "assistant", "content": response})

            # Keep chat history manageable (last 6 turns = 3 pairs)
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]

            rep = trigram_rep(response)
            turn_time = time.time() - turn_start

            state_dict = {DIMS[j]: round(float(state[j]), 4) for j in range(5)}
            pressure_dict = {DIMS[j]: round(float(pressure[j]), 4) for j in range(5)}

            turns_data.append({
                "turn": ti + 1,
                "user": user_msg,
                "response": response,
                "trigram_rep": round(rep, 4),
                "pressure": pressure_dict,
                "state": state_dict,
                "clipped": clipped,
                "gen_time": round(turn_time, 2),
            })
            trajectory.append(state.copy().tolist())

            state_str = " ".join(f"{DIM_SHORT[j]}:{state[j]:+.2f}" for j in range(5))
            log(f"  T{ti+1:02d} [{state_str}] rep={rep:.3f} {turn_time:.1f}s")
            if ti < 2 or ti == len(scenario["turns"]) - 1:
                log(f"       {response[:120]}{'...' if len(response)>120 else ''}")

        session_time = time.time() - session_start

        # Session metrics
        final_state = state.copy()
        drift_magnitude = float(np.linalg.norm(final_state))
        max_rep = max(t["trigram_rep"] for t in turns_data)
        avg_rep = np.mean([t["trigram_rep"] for t in turns_data])
        n_clipped = sum(1 for t in turns_data if t["clipped"])

        session_data = {
            "session_id": scenario["id"],
            "theme": scenario["theme"],
            "n_turns": len(scenario["turns"]),
            "trajectory": trajectory,
            "turns": turns_data,
            "metrics": {
                "drift_magnitude": round(drift_magnitude, 4),
                "final_state": {DIMS[j]: round(float(final_state[j]), 4) for j in range(5)},
                "max_trigram_rep": round(float(max_rep), 4),
                "avg_trigram_rep": round(float(avg_rep), 4),
                "envelope_clips": n_clipped,
                "session_time_s": round(session_time, 1),
            },
        }
        all_sessions.append(session_data)

        log(f"  Done: drift={drift_magnitude:.3f} max_rep={max_rep:.3f} "
            f"clips={n_clipped} time={session_time:.0f}s")

        # Save intermediate progress
        if (si + 1) % 5 == 0 or si == len(SCENARIOS) - 1:
            _save_results(all_sessions, mode, output_path, t_start)

    total_time = time.time() - t_start
    _save_results(all_sessions, mode, output_path, t_start)

    # Cleanup
    ctrl_model.unwrap()
    for h in sde_handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    log(f"\n{'='*60}")
    log(f"COMPLETE: {len(all_sessions)} sessions, {total_time/60:.1f} min")
    log(f"Output: {output_path}")


def _save_results(sessions, mode, output_path, t_start):
    total_turns = sum(s["n_turns"] for s in sessions)
    avg_drift = np.mean([s["metrics"]["drift_magnitude"] for s in sessions])
    avg_rep = np.mean([s["metrics"]["avg_trigram_rep"] for s in sessions])

    result = {
        "experiment": "joi_longrun_drift",
        "mode": mode,
        "model": MODEL_PATH,
        "sde_config": {"targets": SDE_TARGETS, "scale": SDE_SCALE} if mode == "sde" else None,
        "drift_params": {"eta": ETA, "momentum": MOMENTUM, "projection_layer": PROJECTION_LAYER},
        "envelope": ENVELOPE,
        "summary": {
            "n_sessions": len(sessions),
            "total_turns": total_turns,
            "avg_drift_magnitude": round(float(avg_drift), 4),
            "avg_trigram_rep": round(float(avg_rep), 4),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        },
        "sessions": sessions,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"  [saved: {len(sessions)} sessions → {output_path}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "sde"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args.mode, args.output, args.device)
