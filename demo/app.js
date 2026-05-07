const examples = {
  mandarin: {
    kind: "Zero-shot podcast",
    title: "Mandarin podcast dialogue",
    status: "Base sample",
    audio: "audio/mandarin-podcast.wav",
    lines: [
      [
        "S1",
        "哈喽，AI时代的冲浪先锋们！欢迎收听《AI生活进行时》。啊，一个充满了未来感，然后，还有一点点，<|laughter|>神经质的播客节目，我是主持人小希。",
      ],
      ["S2", "哎，大家好呀！我是能唠，爱唠，天天都想唠的唠嗑！"],
      [
        "S1",
        "最近活得特别赛博朋克哈！以前老是觉得AI是科幻片儿里的，<|sigh|> 现在，现在连我妈都用AI写广场舞文案了。",
      ],
      [
        "S2",
        "这个例子很生动啊。是的，特别是生成式AI哈，感觉都要炸了！诶，那我们今天就聊聊AI是怎么走进我们的生活的哈！",
      ],
    ],
  },
  sichuan: {
    kind: "Cross-dialect transfer",
    title: "Sichuanese podcast dialogue",
    status: "Dialect sample",
    audio: "audio/sichuan-transfer.wav",
    lines: [
      [
        "S1",
        "<|Sichuan|>各位《巴适得板》的听众些，大家好噻！我是你们主持人晶晶。今儿天气硬是巴适，不晓得大家是在赶路嘛，还是茶都泡起咯，准备跟我们好生摆一哈龙门阵喃？",
      ],
      [
        "S2",
        "<|Sichuan|>晶晶好哦，大家安逸噻！我是李老倌。你刚开口就川味十足，摆龙门阵几个字一甩出来，我鼻子头都闻到茶香跟火锅香咯！",
      ],
      [
        "S1",
        "<|Sichuan|>就是得嘛！李老倌，我前些天带个外地朋友切人民公园鹤鸣茶社坐了一哈。他硬是搞不醒豁，为啥子我们一堆人围到杯茶就可以吹一下午壳子。",
      ],
      [
        "S2",
        "<|Sichuan|>摆龙门阵哪是摸鱼嘛，这是我们川渝人特有的交际方式，更是一种活法。今天我们就要好生摆一哈，为啥子四川人活得这么舒坦。",
      ],
    ],
  },
  cantonese: {
    kind: "Cross-dialect transfer",
    title: "Cantonese podcast dialogue",
    status: "Dialect sample",
    audio: "audio/cantonese-transfer.wav",
    lines: [
      [
        "S1",
        "<|Yue|>哈囉大家好啊，歡迎收聽我哋嘅節目。喂，我今日想問你樣嘢啊，你覺唔覺得，嗯，而家揸電動車，最煩，最煩嘅一樣嘢係咩啊？",
      ],
      [
        "S2",
        "<|Yue|>梗係充電啦。大佬啊，搵個位都已經好煩，搵到個位仲要喺度等，你話快極都要半個鐘一個鐘，真係，有時諗起都覺得好冇癮。",
      ],
      [
        "S1",
        "<|Yue|>係咪先。如果我而家同你講，充電可以快到同入油差唔多時間，你信唔信先？喂你平時喺油站入滿一缸油，要幾耐啊？",
      ],
      ["S2", "<|Yue|>差唔多啦，七八分鐘，點都走得啦。電車喎，可以做到咁快？你咪玩啦。"],
    ],
  },
  henan: {
    kind: "Cross-dialect transfer",
    title: "Henanese podcast dialogue",
    status: "Dialect sample",
    audio: "audio/henan-transfer.wav",
    lines: [
      [
        "S1",
        "<|Henan|>哎，大家好啊，欢迎收听咱这一期嘞《瞎聊呗，就这么说》，我是恁嘞老朋友，燕子。",
      ],
      [
        "S2",
        "<|Henan|>大家好，我是老张。燕子啊，今儿瞅瞅你这个劲儿，咋着，是有啥可得劲嘞事儿想跟咱唠唠？",
      ],
      [
        "S1",
        "<|Henan|>最近我刷手机，老是刷住些可逗嘞方言视频，特别是咱河南话，一听我都憋不住笑，咋说嘞，得劲儿哩很。",
      ],
      [
        "S2",
        "<|Henan|>你这回可算说到根儿上了！河南话可不光是说话，它脊梁骨后头藏嘞，是咱一整套鲜活嘞过法儿。",
      ],
    ],
  },
};

const tabButtons = document.querySelectorAll(".tab-button");
const kindEl = document.querySelector("#example-kind");
const titleEl = document.querySelector("#example-title");
const statusEl = document.querySelector("#example-status");
const audioEl = document.querySelector("#example-audio");
const audioFileEl = document.querySelector("#audio-file");
const audioStateEl = document.querySelector("#audio-state");
const scriptLinesEl = document.querySelector("#script-lines");
const copyButton = document.querySelector("#copy-script");

let currentKey = "mandarin";

function highlightTags(text) {
  return text.replace(/(&lt;\|[^|]+\|&gt;|<\|[^|]+\|>)/g, (match) => {
    const escaped = match.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `<span class="tag">${escaped}</span>`;
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scriptAsText(example) {
  return example.lines.map(([speaker, text]) => `[${speaker}] ${text}`).join("\n");
}

function renderExample(key) {
  const example = examples[key];
  currentKey = key;

  kindEl.textContent = example.kind;
  titleEl.textContent = example.title;
  statusEl.textContent = example.status;
  audioFileEl.textContent = `Audio file: demo/${example.audio}`;
  audioStateEl.textContent = "Ready";
  audioStateEl.classList.remove("error");

  audioEl.pause();
  audioEl.innerHTML = "";
  const source = document.createElement("source");
  source.src = example.audio;
  source.type = "audio/wav";
  audioEl.appendChild(source);
  audioEl.load();

  scriptLinesEl.innerHTML = example.lines
    .map(([speaker, text]) => {
      const escapedText = escapeHtml(text);
      return `
        <div class="script-line">
          <span class="speaker">${speaker}</span>
          <span class="utterance">${highlightTags(escapedText)}</span>
        </div>
      `;
    })
    .join("");
}

audioEl.addEventListener("canplay", () => {
  audioStateEl.textContent = "Playable";
  audioStateEl.classList.remove("error");
});

audioEl.addEventListener("error", () => {
  audioStateEl.textContent = "Missing file";
  audioStateEl.classList.add("error");
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderExample(button.dataset.example);
  });
});

copyButton.addEventListener("click", async () => {
  const text = scriptAsText(examples[currentKey]);

  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
  } catch {
    copyButton.textContent = "Copy failed";
  }

  window.setTimeout(() => {
    copyButton.textContent = "Copy script";
  }, 1400);
});

renderExample(currentKey);
