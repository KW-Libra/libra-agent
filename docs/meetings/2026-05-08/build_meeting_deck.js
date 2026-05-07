const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const root = __dirname;
const assetsDir = path.join(root, "assets");
const previewDir = path.join(root, "preview");
const heroImageName = "libra-hero-imagegen.png";
fs.mkdirSync(assetsDir, { recursive: true });
fs.mkdirSync(previewDir, { recursive: true });

const W = 1600;
const H = 900;
const pptW = 13.333;
const pptH = 7.5;
const ShapeType = {
  rect: "rect",
  ellipse: "ellipse",
  roundRect: "roundRect",
  line: "line",
};

const colors = {
  paper: "#F7F4EC",
  ink: "#102026",
  muted: "#5F6B6D",
  teal: "#0FA3B1",
  lime: "#C8E24A",
  amber: "#FFB703",
  coral: "#EF476F",
  navy: "#15324A",
  cloud: "#E8EDF0",
  white: "#FFFFFF",
};

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function wrap(text, maxChars) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    if (!line) {
      line = word;
    } else if ((line + " " + word).length <= maxChars) {
      line += " " + word;
    } else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function text(x, y, value, options = {}) {
  const size = options.size || 34;
  const fill = options.fill || colors.ink;
  const weight = options.weight || 500;
  const anchor = options.anchor || "start";
  const maxChars = options.maxChars || 24;
  const lineHeight = options.lineHeight || Math.round(size * 1.25);
  const lines = options.lines || wrap(value, maxChars);
  const tspans = lines
    .map((line, index) => {
      const dy = index === 0 ? 0 : lineHeight;
      return `<tspan x="${x}" dy="${dy}">${esc(line)}</tspan>`;
    })
    .join("");
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="Malgun Gothic, Arial, sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}">${tspans}</text>`;
}

function rounded(x, y, w, h, r, fill, stroke = "none", sw = 0) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function arrow(x1, y1, x2, y2, color = colors.ink, width = 4) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" stroke-linecap="round" marker-end="url(#arrow)"/>`;
}

function baseSvg(body, bg = colors.paper) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="${colors.ink}"/>
    </marker>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#102026" flood-opacity="0.15"/>
    </filter>
  </defs>
  <rect width="${W}" height="${H}" fill="${bg}"/>
  ${body}
</svg>`;
}

async function writePng(name, svg) {
  const file = path.join(assetsDir, name);
  await sharp(Buffer.from(svg)).png().toFile(file);
  return file;
}

function coverSvg() {
  const cx = 1040;
  const cy = 500;
  const nodes = [
    [810, 520, "포트폴리오 설정", colors.teal],
    [990, 300, "데이터 수집", colors.amber],
    [1235, 330, "AI 판단", colors.coral],
    [1320, 570, "주문 제안", colors.lime],
    [1035, 710, "Decision Trace", colors.navy],
  ];
  const body = `
    <circle cx="${cx}" cy="${cy}" r="145" fill="${colors.white}" filter="url(#softShadow)"/>
    ${text(cx, cy - 20, "LIBRA", { size: 76, weight: 800, anchor: "middle", maxChars: 20 })}
    ${text(cx, cy + 35, "설명 가능한 자동 리밸런싱", { size: 28, weight: 600, fill: colors.muted, anchor: "middle", maxChars: 40 })}
    ${nodes.map(([x, y, label, fill]) => `
      ${arrow(x + (x < cx ? 72 : -72), y + (y < cy ? 35 : -35), cx + (x < cx ? -125 : 125), cy + (y < cy ? -42 : 42), colors.ink, 3)}
      <circle cx="${x}" cy="${y}" r="72" fill="${fill}" filter="url(#softShadow)"/>
      ${text(x, y - 8, label, { size: 24, weight: 800, fill: fill === colors.lime ? colors.ink : colors.white, anchor: "middle", maxChars: 10, lineHeight: 31 })}
    `).join("")}
    ${text(105, 150, "개인 투자자를 위한", { size: 42, weight: 650, fill: colors.muted, maxChars: 30 })}
    ${text(105, 225, "AI 멀티 에이전트 기반", { size: 62, weight: 850, maxChars: 24 })}
    ${text(105, 310, "자동 포트폴리오", { size: 56, weight: 850, fill: colors.teal, maxChars: 18 })}
    ${text(105, 385, "리밸런싱 시스템", { size: 56, weight: 850, fill: colors.teal, maxChars: 18 })}
    ${text(108, 475, "Direct Indexing은 대상 투자 방식이고, LIBRA의 핵심은 설명 가능한 판단 구조다.", { size: 25, weight: 650, fill: colors.muted, maxChars: 35, lineHeight: 34 })}
    ${text(1210, 760, "2026.05.08 지도교수 미팅", { size: 28, weight: 600, fill: colors.muted, maxChars: 30 })}
  `;
  return baseSvg(body);
}

function architectureSvg() {
  const items = [
    ["1", "사용자 UI", "회원가입, 온보딩, 목표비중 설정", colors.teal],
    ["2", "데이터 수집", "KIS 가격/잔고, 뉴스, 공시, 리포트", colors.amber],
    ["3", "AI 판단", "Judge + 관점 에이전트", colors.coral],
    ["4", "리밸런싱 처리", "후보 주문, KIS 모의주문 게이트", colors.lime],
    ["5", "리포팅", "History, Decision Trace, 평가", colors.navy],
  ];
  let body = text(90, 100, "전체 기능 아키텍처", { size: 56, weight: 850, maxChars: 30 });
  body += text(92, 154, "교수님 피드백의 5개 컴포넌트를 실제 데이터 흐름으로 정리", { size: 28, weight: 500, fill: colors.muted, maxChars: 60 });
  items.forEach((item, i) => {
    const x = 90 + i * 300;
    const y = 330 + (i % 2) * 70;
    body += rounded(x, y, 235, 170, 18, colors.white, item[3], 5);
    body += `<circle cx="${x + 45}" cy="${y + 48}" r="27" fill="${item[3]}"/>`;
    body += text(x + 45, y + 58, item[0], { size: 28, weight: 850, fill: item[3] === colors.lime ? colors.ink : colors.white, anchor: "middle", maxChars: 4 });
    body += text(x + 80, y + 58, item[1], { size: 30, weight: 850, maxChars: 12 });
    body += text(x + 30, y + 105, item[2], { size: 20, weight: 500, fill: colors.muted, maxChars: 15, lineHeight: 28 });
    if (i < items.length - 1) body += arrow(x + 245, y + 85, x + 290, 415, colors.ink, 4);
  });
  body += text(155, 720, "핵심 데이터: 목표비중 → 현재잔고/가격 → 정규화 신호 → Judge 판단 → 후보 주문 → 판단 이력", { size: 30, weight: 700, fill: colors.ink, maxChars: 80 });
  return baseSvg(body);
}

function multiAgentSvg() {
  const agents = [
    [305, 280, "공시", "기업 이벤트", colors.teal],
    [300, 610, "뉴스", "시장 반응", colors.amber],
    [745, 705, "리포트", "컨센서스", colors.navy],
    [1190, 610, "수익", "기대 방향", colors.coral],
    [1195, 280, "비용", "거래 마찰", colors.lime],
  ];
  let body = text(80, 92, "멀티 에이전트 판단 구조", { size: 56, weight: 850, maxChars: 32 });
  body += text(82, 148, "각 관점이 독립 의견을 내고 Judge가 충돌을 조정한다", { size: 28, fill: colors.muted, maxChars: 56 });
  body += `<circle cx="760" cy="455" r="145" fill="${colors.white}" stroke="${colors.ink}" stroke-width="5" filter="url(#softShadow)"/>`;
  body += text(760, 435, "Judge Agent", { size: 42, weight: 850, anchor: "middle", maxChars: 18 });
  body += text(760, 490, "호출 선택 · 근거 조정 · 최종 판단", { size: 22, weight: 600, fill: colors.muted, anchor: "middle", maxChars: 30 });
  agents.forEach(([x, y, name, desc, fill]) => {
    body += arrow(x + (x < 760 ? 105 : -105), y, 760 + (x < 760 ? -130 : 130), 455 + (y < 455 ? -55 : 55), colors.ink, 3);
    body += rounded(x - 105, y - 55, 210, 110, 18, fill, "none", 0);
    body += text(x, y - 8, name, { size: 34, weight: 850, fill: fill === colors.lime ? colors.ink : colors.white, anchor: "middle", maxChars: 8 });
    body += text(x, y + 28, desc, { size: 20, weight: 600, fill: fill === colors.lime ? colors.ink : colors.white, anchor: "middle", maxChars: 12 });
  });
  body += rounded(180, 785, 1240, 48, 24, colors.cloud);
  body += text(800, 818, "Decision Trace: 어떤 에이전트를 왜 호출했고, 왜 건너뛰었는지까지 기록", { size: 26, weight: 750, anchor: "middle", maxChars: 70 });
  return baseSvg(body);
}

function demoFlowSvg() {
  const steps = [
    ["1", "회원가입/로그인"],
    ["2", "KIS 종목 선택"],
    ["3", "목표비중 저장"],
    ["4", "잔고/가격 동기화"],
    ["5", "Judge Run"],
    ["6", "Trace 확인"],
    ["7", "현재가 수량 산출"],
    ["8", "모의주문 게이트"],
  ];
  let body = text(80, 92, "데모 흐름", { size: 56, weight: 850, maxChars: 30 });
  body += text(82, 148, "보여줄 것은 화면 수가 아니라 end-to-end 판단 경로", { size: 28, fill: colors.muted, maxChars: 60 });
  steps.forEach(([num, label], i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = 110 + col * 370;
    const y = 300 + row * 220;
    const fill = [colors.teal, colors.amber, colors.coral, colors.lime][col];
    body += rounded(x, y, 260, 120, 16, colors.white, fill, 5);
    body += `<circle cx="${x + 45}" cy="${y + 60}" r="30" fill="${fill}"/>`;
    body += text(x + 45, y + 70, num, { size: 28, weight: 850, fill: fill === colors.lime ? colors.ink : colors.white, anchor: "middle", maxChars: 3 });
    body += text(x + 90, y + 68, label, { size: 25, weight: 800, maxChars: 12, lineHeight: 31 });
    if (i < steps.length - 1 && col !== 3) body += arrow(x + 265, y + 60, x + 345, y + 60, colors.ink, 4);
    if (i === 3) body += arrow(x + 130, y + 130, 240, 520, colors.ink, 4);
  });
  return baseSvg(body);
}

function benchmarkSvg() {
  const items = [
    ["Buy & Hold", "초기 매수 후 방치", colors.teal],
    ["기계적 리밸런싱", "비중 이탈만 기준", colors.amber],
    ["LIBRA", "비중 + 신호 + 비용 + trace", colors.coral],
  ];
  let body = text(80, 92, "검증 설계", { size: 56, weight: 850, maxChars: 30 });
  body += text(82, 148, "수익률을 주장하기 전에 비교군과 지표를 고정한다", { size: 28, fill: colors.muted, maxChars: 60 });
  items.forEach(([title, desc, fill], i) => {
    const x = 150 + i * 440;
    body += rounded(x, 270, 340, 260, 24, colors.white, fill, 6);
    body += text(x + 170, 350, title, { size: 36, weight: 850, anchor: "middle", maxChars: 14 });
    body += text(x + 170, 410, desc, { size: 24, weight: 600, fill: colors.muted, anchor: "middle", maxChars: 18, lineHeight: 32 });
  });
  body += rounded(170, 650, 1260, 92, 46, colors.ink);
  body += text(800, 707, "비용 차감 수익률 · MDD · 거래횟수 · 회전율 · 판단 trace 완성도", { size: 32, weight: 800, fill: colors.white, anchor: "middle", maxChars: 74 });
  return baseSvg(body);
}

function slidePreviewSvg(title, subtitle, bullets = [], accent = colors.teal, imageName = null) {
  let body = text(82, 105, title, { size: 58, weight: 850, maxChars: 34 });
  if (subtitle) body += text(86, 165, subtitle, { size: 28, fill: colors.muted, maxChars: 70 });
  if (imageName) {
    const imagePath = path.join(assetsDir, imageName);
    const data = fs.readFileSync(imagePath).toString("base64");
    body += `<image href="data:image/png;base64,${data}" x="115" y="220" width="1370" height="610" preserveAspectRatio="xMidYMid meet"/>`;
  } else {
    bullets.forEach((b, i) => {
      const y = 275 + i * 92;
      body += `<circle cx="135" cy="${y - 10}" r="14" fill="${accent}"/>`;
      body += text(170, y, b, { size: 31, weight: 650, maxChars: 56, lineHeight: 40 });
    });
  }
  return baseSvg(body);
}

async function buildAssets() {
  await writePng("cover-system-map.png", coverSvg());
  await writePng("functional-architecture.png", architectureSvg());
  await writePng("multi-agent-judge.png", multiAgentSvg());
  await writePng("demo-flow.png", demoFlowSvg());
  await writePng("benchmark-plan.png", benchmarkSvg());
}

function addTitle(slide, title, subtitle) {
  slide.background = { color: "F7F4EC" };
  slide.addText(title, { x: 0.72, y: 0.42, w: 11.8, h: 0.55, fontFace: "Malgun Gothic", fontSize: 27, bold: true, color: "102026", margin: 0 });
  if (subtitle) slide.addText(subtitle, { x: 0.74, y: 0.98, w: 11.8, h: 0.35, fontFace: "Malgun Gothic", fontSize: 13.5, color: "5F6B6D", margin: 0 });
}

function addBullets(slide, items, x = 0.9, y = 1.75, w = 11.3) {
  items.forEach((item, i) => {
    slide.addShape(ShapeType.ellipse, { x, y: y + i * 0.72 + 0.08, w: 0.12, h: 0.12, fill: { color: item.color || "0FA3B1" }, line: { color: item.color || "0FA3B1" } });
    slide.addText(item.text || item, { x: x + 0.28, y: y + i * 0.72, w, h: 0.38, fontFace: "Malgun Gothic", fontSize: item.size || 16, bold: !!item.bold, color: item.ink || "102026", fit: "shrink", margin: 0 });
  });
}

function addFooter(slide, n) {
  slide.addText(`LIBRA 교수 미팅 준비 | ${n}`, { x: 10.85, y: 7.08, w: 1.8, h: 0.2, fontFace: "Malgun Gothic", fontSize: 8.5, color: "5F6B6D", margin: 0, align: "right" });
}

function addPill(slide, x, y, textValue, color) {
  slide.addShape(ShapeType.roundRect, { x, y, w: 2.0, h: 0.36, rectRadius: 0.08, fill: { color }, line: { color } });
  slide.addText(textValue, { x: x + 0.08, y: y + 0.08, w: 1.84, h: 0.16, fontFace: "Malgun Gothic", fontSize: 10.5, bold: true, color: color === "C8E24A" ? "102026" : "FFFFFF", align: "center", margin: 0 });
}

function addHeroCover(slide, asset) {
  const heroPath = asset(heroImageName);
  if (fs.existsSync(heroPath)) {
    slide.addImage({ path: heroPath, x: 0, y: 0, w: pptW, h: pptH });
  } else {
    slide.addImage({ path: asset("cover-system-map.png"), x: 0, y: 0, w: pptW, h: pptH });
  }

  slide.addShape(ShapeType.rect, { x: 0, y: 0, w: 5.25, h: pptH, fill: { color: "F7F4EC", transparency: 6 }, line: { color: "F7F4EC", transparency: 100 } });
  slide.addShape(ShapeType.rect, { x: 5.25, y: 0, w: 0.75, h: pptH, fill: { color: "F7F4EC", transparency: 45 }, line: { color: "F7F4EC", transparency: 100 } });
  slide.addShape(ShapeType.line, { x: 0.78, y: 0.76, w: 1.05, h: 0, line: { color: "0FA3B1", width: 4 } });
  slide.addText("LIBRA", { x: 0.72, y: 1.04, w: 4.5, h: 0.72, fontFace: "Malgun Gothic", fontSize: 48, bold: true, color: "102026", margin: 0, fit: "shrink" });
  slide.addText("개인 투자자를 위한 설명 가능한 AI 멀티 에이전트 기반 자동 리밸런싱 판단 시스템", {
    x: 0.76,
    y: 1.92,
    w: 4.08,
    h: 1.02,
    fontFace: "Malgun Gothic",
    fontSize: 20,
    bold: true,
    color: "15324A",
    fit: "shrink",
    margin: 0,
    breakLine: false,
  });
  slide.addText("목표비중 → 다중 관점 판단 → Decision Trace → KIS 주문 제안", {
    x: 0.78,
    y: 3.26,
    w: 4.0,
    h: 0.58,
    fontFace: "Malgun Gothic",
    fontSize: 15.5,
    color: "102026",
    bold: true,
    fit: "shrink",
    margin: 0,
  });
  slide.addText("내일 미팅 핵심: 제목, 문제정의, 요구사항, 아키텍처, 검증 계획", {
    x: 0.78,
    y: 6.42,
    w: 4.2,
    h: 0.36,
    fontFace: "Malgun Gothic",
    fontSize: 12.5,
    color: "5F6B6D",
    fit: "shrink",
    margin: 0,
  });
}

function heroCoverOverlaySvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    <rect x="0" y="0" width="630" height="${H}" fill="#F7F4EC" fill-opacity="0.94"/>
    <rect x="630" y="0" width="95" height="${H}" fill="#F7F4EC" fill-opacity="0.55"/>
    <line x1="94" y1="91" x2="220" y2="91" stroke="#0FA3B1" stroke-width="5"/>
    ${text(88, 172, "LIBRA", { size: 66, weight: 800, fill: colors.ink, maxChars: 18 })}
    ${text(92, 252, "개인 투자자를 위한 설명 가능한 AI 멀티 에이전트 기반 자동 리밸런싱 판단 시스템", { size: 30, weight: 800, fill: colors.navy, maxChars: 20, lineHeight: 41 })}
    ${text(94, 416, "목표비중 → 다중 관점 판단 → Decision Trace → KIS 주문 제안", { size: 22, weight: 750, fill: colors.ink, maxChars: 24, lineHeight: 33 })}
    ${text(94, 790, "내일 미팅 핵심: 제목, 문제정의, 요구사항, 아키텍처, 검증 계획", { size: 18, weight: 600, fill: colors.muted, maxChars: 28, lineHeight: 27 })}
  </svg>`;
}

async function writeHeroCoverPreview() {
  const heroPath = path.join(assetsDir, heroImageName);
  const out = path.join(previewDir, "slide-01-cover.png");
  if (!fs.existsSync(heroPath)) {
    await sharp(Buffer.from(coverSvg())).png().toFile(out);
    return;
  }
  await sharp(heroPath)
    .resize(W, H, { fit: "cover" })
    .composite([{ input: Buffer.from(heroCoverOverlaySvg()), left: 0, top: 0 }])
    .png()
    .toFile(out);
}

async function buildDeck() {
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "LIBRA";
  pptx.subject = "2026-05-08 professor meeting";
  pptx.title = "LIBRA 미팅 준비 자료";
  pptx.company = "KW-Libra";
  pptx.lang = "ko-KR";
  pptx.theme = {
    headFontFace: "Malgun Gothic",
    bodyFontFace: "Malgun Gothic",
    lang: "ko-KR",
  };

  const asset = (name) => path.join(assetsDir, name);

  let slide = pptx.addSlide();
  slide.background = { color: "F7F4EC" };
  addHeroCover(slide, asset);

  slide = pptx.addSlide();
  addTitle(slide, "오늘 가져갈 답", "교수님 피드백을 자료 구조와 실제 구현 흐름으로 다시 정리");
  addBullets(slide, [
    { text: "제목: Direct Indexing이 아니라 사용자가 이해하는 시스템명으로 재정의", bold: true, color: "0FA3B1" },
    { text: "문제정의: 포트폴리오 투자 → 개인화 → 운용 난점 → AI 리밸런싱으로 연결", color: "FFB703" },
    { text: "요구사항: 회원가입, KIS 종목 선택, 목표비중, 잔고 동기화, AI 판단, 주문 제안까지 번호로 명세", color: "EF476F" },
    { text: "시연: 목표비중과 현재비중 차이로 REBALANCE 판단과 KIS 현재가 주문 제안 확인", color: "C8E24A" },
  ]);
  addFooter(slide, 2);

  slide = pptx.addSlide();
  addTitle(slide, "문제정의 스토리라인", "한 장의 문단처럼 연결되게 말한다");
  const story = [
    ["그래서", "포트폴리오 투자는 단일 종목 리스크를 줄이는 방법이다.", "0FA3B1"],
    ["하지만", "개인은 ETF가 아니라 자기 철학에 맞는 종목 조합을 원한다.", "FFB703"],
    ["그렇기 때문에", "직접 고른 종목은 비중 조정, 뉴스 반영, 비용 판단이 필요하다.", "EF476F"],
    ["종합하면", "LIBRA는 이 과정을 설명 가능한 AI 판단 흐름으로 자동화한다.", "15324A"],
  ];
  story.forEach((item, i) => {
    const x = 0.9 + i * 3.05;
    slide.addShape(ShapeType.roundRect, { x, y: 2.05, w: 2.55, h: 2.85, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: item[2], width: 2 } });
    slide.addText(item[0], { x: x + 0.16, y: 2.28, w: 2.2, h: 0.34, fontFace: "Malgun Gothic", fontSize: 19, bold: true, color: item[2], margin: 0 });
    slide.addText(item[1], { x: x + 0.16, y: 2.93, w: 2.23, h: 1.15, fontFace: "Malgun Gothic", fontSize: 15, color: "102026", fit: "shrink", breakLine: false, margin: 0.02 });
  });
  slide.addText("이 흐름이 끊기면 또 “그래서 뭘 만든다는 거냐”로 돌아간다.", { x: 1.15, y: 5.7, w: 10.8, h: 0.42, fontFace: "Malgun Gothic", fontSize: 18, bold: true, color: "102026", align: "center", margin: 0 });
  addFooter(slide, 3);

  slide = pptx.addSlide();
  addTitle(slide, "요구사항 명세", "빙빙 돌려 설명하지 않고 만들 기능을 번호로 고정");
  const reqs = [
    "회원가입/로그인과 사용자별 데이터 분리",
    "KIS 종목 조회와 목표비중 설정",
    "KIS 잔고·현재가 동기화",
    "목표비중 대비 현재비중 드리프트 계산",
    "공시·뉴스·리포트·수익·비용 에이전트 판단",
    "Decision Trace와 판단 이력 저장",
    "KIS 현재가 기반 모의주문 수량 산출",
    "Buy & Hold / 기계적 리밸런싱 / LIBRA 비교 검증",
  ];
  reqs.forEach((r, i) => {
    const col = i < 4 ? 0 : 1;
    const x = col === 0 ? 0.95 : 6.95;
    const y = 1.75 + (i % 4) * 0.82;
    slide.addText(`${i + 1}`, { x, y, w: 0.36, h: 0.32, fontFace: "Malgun Gothic", fontSize: 13, bold: true, color: "FFFFFF", align: "center", margin: 0.02, fill: { color: i % 2 ? "EF476F" : "0FA3B1" } });
    slide.addText(r, { x: x + 0.55, y: y - 0.01, w: 5.0, h: 0.36, fontFace: "Malgun Gothic", fontSize: 15.5, color: "102026", fit: "shrink", margin: 0 });
  });
  addFooter(slide, 4);

  slide = pptx.addSlide();
  slide.background = { color: "F7F4EC" };
  slide.addImage({ path: asset("functional-architecture.png"), x: 0, y: 0, w: pptW, h: pptH });
  addFooter(slide, 5);

  slide = pptx.addSlide();
  slide.background = { color: "F7F4EC" };
  slide.addImage({ path: asset("multi-agent-judge.png"), x: 0, y: 0, w: pptW, h: pptH });
  addFooter(slide, 6);

  slide = pptx.addSlide();
  addTitle(slide, "정성 데이터 정량화", "뉴스·공시·리포트를 Judge가 비교 가능한 공통 신호로 바꾼다");
  const fields = [
    ["direction", "긍정/부정 방향"],
    ["strength", "신호 강도"],
    ["confidence", "판단 신뢰도"],
    ["source_trust", "출처 신뢰도"],
    ["horizon", "영향 기간"],
    ["risk_level", "위험 수준"],
    ["reasoning", "Judge 전달 근거"],
  ];
  fields.forEach(([k, v], i) => {
    const x = i % 2 === 0 ? 1.1 : 7.1;
    const y = 1.8 + Math.floor(i / 2) * 0.72;
    addPill(slide, x, y, k, i % 3 === 0 ? "0FA3B1" : i % 3 === 1 ? "EF476F" : "FFB703");
    slide.addText(v, { x: x + 2.25, y: y + 0.04, w: 3.2, h: 0.24, fontFace: "Malgun Gothic", fontSize: 14.2, color: "102026", margin: 0 });
  });
  slide.addText("핵심 방어 논리", { x: 1.1, y: 5.42, w: 2.2, h: 0.3, fontFace: "Malgun Gothic", fontSize: 17, bold: true, color: "102026", margin: 0 });
  slide.addText("LLM 자연어를 그대로 믿는 것이 아니라, 공통 스키마로 정규화하고 스키마 검증을 거친 뒤 Judge 판단에 넣는다.", { x: 1.1, y: 5.88, w: 10.8, h: 0.55, fontFace: "Malgun Gothic", fontSize: 18, bold: true, color: "15324A", fit: "shrink", margin: 0 });
  addFooter(slide, 7);

  slide = pptx.addSlide();
  addTitle(slide, "현재 구현된 end-to-end 흐름", "오늘 로컬에서 확인한 실제 동작 증거");
  addBullets(slide, [
    { text: "목표 포트폴리오: 삼성전자 50%, SK하이닉스 50%", bold: true, color: "0FA3B1" },
    { text: "현재 포트폴리오: 현금 100% 또는 KIS 동기화 잔고", color: "FFB703" },
    { text: "Agent 판단: REBALANCE, 후보 초안 005930 +0.1 / 000660 +0.1", color: "EF476F" },
    { text: "Decision Trace: Judge가 호출한 에이전트와 판단 근거 저장", color: "15324A" },
    { text: "KIS 현재가 기반 주문 제안: 2건 생성, 예시 삼성전자 BUY 11주", color: "C8E24A" },
  ], 0.9, 1.72, 11.2);
  addFooter(slide, 8);

  slide = pptx.addSlide();
  slide.background = { color: "F7F4EC" };
  slide.addImage({ path: asset("demo-flow.png"), x: 0, y: 0, w: pptW, h: pptH });
  addFooter(slide, 9);

  slide = pptx.addSlide();
  slide.background = { color: "F7F4EC" };
  slide.addImage({ path: asset("benchmark-plan.png"), x: 0, y: 0, w: pptW, h: pptH });
  addFooter(slide, 10);

  slide = pptx.addSlide();
  addTitle(slide, "팀 역할과 통합 방향", "팀원 repo를 5개 컴포넌트에 배치해서 설명");
  const rows = [
    ["데이터 수집", "ingest / collector", "뉴스·공시·리포트 정규화"],
    ["AI 판단", "agent repos", "관점 에이전트로 흡수"],
    ["리밸런싱 처리", "backend / libra-direct", "KIS 주문 후보와 모의주문"],
    ["리포팅", "backend + frontend", "History, Trace, 평가"],
    ["UI", "frontend", "온보딩, 설정, 실행 게이트"],
  ];
  rows.forEach((r, i) => {
    const y = 1.72 + i * 0.82;
    slide.addText(r[0], { x: 0.95, y, w: 2.05, h: 0.32, fontFace: "Malgun Gothic", fontSize: 15, bold: true, color: "102026", margin: 0 });
    slide.addText(r[1], { x: 3.35, y, w: 2.55, h: 0.32, fontFace: "Malgun Gothic", fontSize: 14, color: "0FA3B1", bold: true, margin: 0 });
    slide.addText(r[2], { x: 6.3, y, w: 5.5, h: 0.32, fontFace: "Malgun Gothic", fontSize: 14.4, color: "102026", margin: 0 });
    slide.addShape(ShapeType.line, { x: 0.95, y: y + 0.48, w: 10.9, h: 0, line: { color: "D6D9D8", width: 1 } });
  });
  slide.addText("내일 표현: “완전 통합 완료”가 아니라 “컴포넌트별 통합 기준과 반영 계획을 잡았다.”", { x: 1.05, y: 6.25, w: 10.9, h: 0.4, fontFace: "Malgun Gothic", fontSize: 17, bold: true, color: "15324A", align: "center", margin: 0 });
  addFooter(slide, 11);

  slide = pptx.addSlide();
  addTitle(slide, "내일의 결론", "완성 선언보다 중요한 것은 문제정의와 검증 가능한 구현 흐름");
  addBullets(slide, [
    { text: "LIBRA의 목표는 설명 가능한 AI 판단 기반 자동 리밸런싱 시스템으로 고정", bold: true, color: "0FA3B1" },
    { text: "핵심 end-to-end 흐름은 실제로 연결됨: 목표비중 → Judge → Trace → KIS 주문 제안", color: "EF476F" },
    { text: "다음 고도화는 Claude 기반 수익/비용 판단, 온보딩 UX, 백테스트 검증", color: "FFB703" },
    { text: "교수님께 요청할 피드백: 정성 데이터 정량화와 성능 지표 정의가 타당한지", color: "15324A" },
  ], 0.9, 1.8, 11.1);
  slide.addText("한 문장으로 말하기", { x: 1.0, y: 5.75, w: 2.2, h: 0.3, fontFace: "Malgun Gothic", fontSize: 16, bold: true, color: "5F6B6D", margin: 0 });
  slide.addText("“이제 무엇을 만들지 불명확한 상태가 아니라, 기능 명세와 판단 구조를 기준으로 구현과 검증을 진행하고 있습니다.”", { x: 1.0, y: 6.18, w: 11.2, h: 0.5, fontFace: "Malgun Gothic", fontSize: 19, bold: true, color: "102026", fit: "shrink", margin: 0 });
  addFooter(slide, 12);

  const preferredOut = path.join(root, "LIBRA_2026-05-08_professor_meeting.pptx");
  const lockFile = path.join(root, "~$LIBRA_2026-05-08_professor_meeting.pptx");
  const out = fs.existsSync(lockFile)
    ? path.join(root, "LIBRA_2026-05-08_professor_meeting_imagegen.pptx")
    : preferredOut;
  await pptx.writeFile({ fileName: out });
  return out;
}

async function buildPreviews() {
  const previews = [
    ["slide-02-answer.png", slidePreviewSvg("오늘 가져갈 답", "교수님 피드백을 자료 구조와 실제 구현 흐름으로 다시 정리", ["제목 재정의", "문제정의 재구성", "요구사항 번호 명세", "동작 증거와 검증 계획"], colors.teal)],
    ["slide-03-story.png", slidePreviewSvg("문제정의 스토리라인", "포트폴리오 투자 → 개인화 → 운용 난점 → AI 리밸런싱", ["그래서: 포트폴리오 투자는 필요하다", "하지만: 개인은 자기 종목 조합을 원한다", "그렇기 때문에: 비중·뉴스·비용 판단이 어렵다", "종합하면: LIBRA가 판단 흐름을 자동화한다"], colors.amber)],
    ["slide-04-requirements.png", slidePreviewSvg("요구사항 명세", "만들 기능을 번호로 고정", ["회원가입/로그인, 사용자별 데이터 분리", "KIS 종목 조회와 목표비중 설정", "드리프트 계산과 AI 판단", "Decision Trace와 KIS 주문 제안"], colors.coral)],
    ["slide-05-architecture.png", architectureSvg()],
    ["slide-06-agents.png", multiAgentSvg()],
    ["slide-07-signals.png", slidePreviewSvg("정성 데이터 정량화", "뉴스·공시·리포트를 공통 신호로 변환", ["direction, strength, confidence", "source_trust, horizon, risk_level", "reasoning_for_judge_agent", "스키마 검증과 정규화"], colors.teal)],
    ["slide-08-evidence.png", slidePreviewSvg("현재 구현된 흐름", "목표비중 → Judge → Trace → KIS 주문 제안", ["REBALANCE 판단 확인", "후보 초안 005930/000660 +0.1", "Decision Trace 저장", "KIS 현재가 기반 주문 제안 2건"], colors.coral)],
    ["slide-09-demo.png", demoFlowSvg()],
    ["slide-10-benchmark.png", benchmarkSvg()],
    ["slide-11-team.png", slidePreviewSvg("팀 역할과 통합 방향", "팀원 repo를 5개 컴포넌트에 배치", ["데이터 수집: ingest/collector", "AI 판단: 관점 에이전트", "리밸런싱 처리: KIS 주문", "UI/리포팅: History와 Trace"], colors.navy)],
    ["slide-12-close.png", slidePreviewSvg("내일의 결론", "문제정의와 검증 가능한 구현 흐름", ["목표는 설명 가능한 AI 리밸런싱 시스템", "핵심 end-to-end 흐름은 연결됨", "다음은 Claude 고도화와 백테스트", "정성 데이터 정량화 피드백 요청"], colors.lime)],
  ];
  await writeHeroCoverPreview();
  for (const [name, svg] of previews) {
    await sharp(Buffer.from(svg)).png().toFile(path.join(previewDir, name));
  }
  const previewNames = ["slide-01-cover.png", ...previews.map(([name]) => name)];
  const thumbs = [];
  for (const name of previewNames) {
    const input = await sharp(path.join(previewDir, name)).resize(400, 225).png().toBuffer();
    thumbs.push({ input, left: (thumbs.length % 3) * 400, top: Math.floor(thumbs.length / 3) * 225 });
  }
  await sharp({
    create: {
      width: 1200,
      height: 900,
      channels: 4,
      background: colors.paper,
    },
  })
    .composite(thumbs)
    .png()
    .toFile(path.join(previewDir, "contact-sheet.png"));
}

async function main() {
  await buildAssets();
  const deck = await buildDeck();
  await buildPreviews();
  console.log(JSON.stringify({
    deck,
    assetsDir,
    previewDir,
    contactSheet: path.join(previewDir, "contact-sheet.png"),
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
