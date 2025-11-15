// ===== 国际化 (i18n) =====
let currentLang = localStorage.getItem('language') || 'zh';

const i18n = {
    zh: {
        'back-to-home': '返回主页',
        'time-label': '时间 Time',
        'del-hint': '删除',
        'esc-hint': '取消',
        'validate-text': '验证守恒',
        'validate-short': '验证',
        'help-title': '物理守恒定律完整版',
        'tool-select': '选择',
        'tool-fermion': '费米子',
        'tool-photon': '光子',
        'tool-gluon': '胶子',
        'tool-higgs': '希格斯',
        'label-family': '粒子族 (Family)',
        'family-lepton': '轻子',
        'family-quark-u': '上型夸克',
        'family-quark-d': '下型夸克',
        'label-particle': '具体粒子',
        'label-color': '色荷 (Color)',
        'color-red': '红 (Red)',
        'color-green': '绿 (Green)',
        'color-blue': '蓝 (Blue)',
        'label-conservation': '守恒量',
        'charge': 'Q(电荷):',
        'lepton': 'L(轻子数):',
        'baryon': 'B(重子数):',
        'spin': '自旋:',
        'parity': '宇称:',
        'label-boson': '玻色子类型',
        'label-gluon-color': '胶子色荷组合',
        'empty-state-text': '请选择工具开始绘制，靠近端点自动吸附。',
        'confirm-clear': '确定清空画布吗？',
        'toast-no-vertex': 'ℹ️ 未发现可验证的顶点 (请确保线条吸附)',
        'toast-validate-pass': '✅ {count} 个顶点验证通过！',
        'toast-validate-warn': '⚠️ {count} 个警告',
        'toast-validate-error': '❌ 发现 {count} 处错误',
        'toast-empty': 'ℹ️ 画布为空',
        'selected-multi': '已选中 {count} 个粒子',
        'empty-select': '请选择一个粒子或工具',
        'boson-w-direction': 'L→R = W⁻, R→L = W⁺',
        // AI 相关
        'ai-generate': 'AI 生成',
        'ai-title': '🤖 AI 自动生成费曼图',
        'api-key-notice': '首次使用需要设置 Gemini API Key',
        'api-key-instruction': '获取方式：访问 Google AI Studio 免费获取',
        'api-key-label': 'Gemini API Key',
        'save-key': '保存',
        'reaction-label': '输入反应式',
        'reaction-examples': '示例：',
        'generate-button': '🎨 生成费曼图',
        'ai-loading-text': 'AI 正在思考中...',
        'ai-success': '生成成功！',
        'help-content': `
            <div class="bg-cyan-900/30 p-2 rounded border border-cyan-700">
                <div class="font-bold text-cyan-300 mb-1 text-xs">🔴 严格守恒定律</div>
                <div class="text-slate-300 text-[11px] space-y-0.5">
                    <div>• 电荷守恒 (Q) - 所有相互作用</div>
                    <div>• 轻子数守恒 (L) - 所有相互作用</div>
                    <div>• 重子数守恒 (B) - 所有相互作用</div>
                    <div>• 能量-动量守恒 - 所有相互作用 ⭐新增</div>
                    <div>• 色荷守恒 - 强相互作用（QCD）⭐增强</div>
                </div>
            </div>
            <div class="bg-emerald-900/30 p-2 rounded border border-emerald-700">
                <div class="font-bold text-emerald-300 mb-1 text-xs">🟢 条件守恒定律</div>
                <div class="text-slate-300 text-[11px] space-y-0.5">
                    <div>• 宇称守恒 - 强/电磁，弱破坏</div>
                    <div>• 味守恒 - 强/电磁，弱破坏 ⭐CKM矩阵</div>
                    <div>• 自旋/角动量守恒 - 所有顶点</div>
                </div>
            </div>
            <div class="bg-purple-900/30 p-2 rounded border border-purple-700">
                <div class="font-bold text-purple-300 mb-1 text-xs">🟣 色荷规则 (QCD)</div>
                <div class="text-slate-300 text-[11px]">
                    夸克携带色荷（红/绿/蓝），反夸克携带反色荷。
                    胶子携带色-反色组合（如rḡ=红+反绿）。顶点必须形成色单态（白色）。
                </div>
            </div>
            <div class="bg-orange-900/30 p-2 rounded border border-orange-700">
                <div class="font-bold text-orange-300 mb-1 text-xs">🟠 相互作用规则 ⭐增强</div>
                <ul class="text-slate-300 text-[11px] space-y-0.5 list-disc list-inside">
                    <li>光子(γ)只与带电粒子相互作用 ⭐强制</li>
                    <li>胶子(g)只与夸克相互作用 ⭐强制</li>
                    <li>W±玻色子可以改变粒子味 ⭐CKM矩阵</li>
                    <li>弱相互作用破坏宇称守恒</li>
                    <li>顶点维度不能超过4.0 ⭐新增</li>
                    <li>耦合常数: αₛ≈0.12, αₑₘ≈1/137, αw≈0.03 ⭐新增</li>
                </ul>
            </div>
            <div class="bg-slate-800 p-2 rounded border border-slate-700">
                <div class="font-bold text-slate-300 mb-1 text-xs">💡 使用提示</div>
                <ul class="text-slate-400 text-[11px] space-y-0.5 list-disc list-inside">
                    <li>费米子：从左到右=正粒子，从右到左=反粒子</li>
                    <li>W玻色子：在面板中选择 W⁺ 或 W⁻</li>
                    <li>夸克需要选择色荷</li>
                    <li>胶子需要选择色荷组合（色-反色）⭐修正</li>
                    <li>蓝色虚线：初态（左）和末态（右）边界 ⭐</li>
                    <li>质量守恒：只对连接外线的顶点检查 ⭐</li>
                    <li>内线（虚粒子）可以违反能量守恒 ⭐</li>
                    <li>CKM矩阵：u↔d最强(97%), u↔b最弱(0.4%) ⭐新增</li>
                    <li>辐射修正：光子/胶子可附着在外线上（e⁻→e⁻+γ）⭐新增</li>
                </ul>
            </div>
        `
    },
    en: {
        'back-to-home': 'Back to Home',
        'time-label': 'Time',
        'del-hint': 'Delete',
        'esc-hint': 'Cancel',
        'validate-text': 'Validate',
        'validate-short': 'Check',
        'help-title': 'Physics Conservation Laws',
        'tool-select': 'Select',
        'tool-fermion': 'Fermion',
        'tool-photon': 'Photon',
        'tool-gluon': 'Gluon',
        'tool-higgs': 'Higgs',
        'label-family': 'Particle Family',
        'family-lepton': 'Lepton',
        'family-quark-u': 'Up-type',
        'family-quark-d': 'Down-type',
        'label-particle': 'Specific Particle',
        'label-color': 'Color Charge',
        'color-red': 'Red',
        'color-green': 'Green',
        'color-blue': 'Blue',
        'label-conservation': 'Conserved Quantities',
        'charge': 'Q (Charge):',
        'lepton': 'L (Lepton):',
        'baryon': 'B (Baryon):',
        'spin': 'Spin:',
        'parity': 'Parity:',
        'label-boson': 'Boson Type',
        'label-gluon-color': 'Gluon Color Combination',
        'empty-state-text': 'Select a tool to start drawing. Auto-snap to endpoints.',
        'confirm-clear': 'Are you sure you want to clear the canvas?',
        'toast-no-vertex': 'ℹ️ No verifiable vertices found (ensure lines snap)',
        'toast-validate-pass': '✅ {count} vertices validated!',
        'toast-validate-warn': '⚠️ {count} warnings',
        'toast-validate-error': '❌ Found {count} errors',
        'toast-empty': 'ℹ️ Canvas is empty',
        'selected-multi': '{count} particles selected',
        'empty-select': 'Select a particle or a tool',
        'boson-w-direction': 'L→R = W⁻, R→L = W⁺',
        'help-content': `
            <div class="bg-cyan-900/30 p-2 rounded border border-cyan-700">
                <div class="font-bold text-cyan-300 mb-1 text-xs">🔴 Strict Conservation</div>
                <div class="text-slate-300 text-[11px] space-y-0.5">
                    <div>• Charge (Q) - All interactions</div>
                    <div>• Lepton number (L) - All interactions</div>
                    <div>• Baryon number (B) - All interactions</div>
                    <div>• Energy-Momentum - All interactions ⭐New</div>
                    <div>• Color charge - Strong (QCD) ⭐Enhanced</div>
                </div>
            </div>
            <div class="bg-emerald-900/30 p-2 rounded border border-emerald-700">
                <div class="font-bold text-emerald-300 mb-1 text-xs">🟢 Conditional Conservation</div>
                <div class="text-slate-300 text-[11px] space-y-0.5">
                    <div>• Parity - Strong/EM, violated by weak</div>
                    <div>• Flavor - Strong/EM, violated by weak ⭐CKM Matrix</div>
                    <div>• Spin/Angular Momentum - All vertices</div>
                </div>
            </div>
            <div class="bg-purple-900/30 p-2 rounded border border-purple-700">
                <div class="font-bold text-purple-300 mb-1 text-xs">🟣 Color Charge Rules</div>
                <div class="text-slate-300 text-[11px]">
                    Quarks carry color (red/green/blue), antiquarks carry anti-color.
                    Gluons carry color-anticolor combinations (e.g., rḡ = red + anti-green). Vertices must form color singlet (white).
                </div>
            </div>
            <div class="bg-orange-900/30 p-2 rounded border border-orange-700">
                <div class="font-bold text-orange-300 mb-1 text-xs">🟠 Interaction Rules ⭐Enhanced</div>
                <ul class="text-slate-300 text-[11px] space-y-0.5 list-disc list-inside">
                    <li>Photon (γ) only with charged particles ⭐Enforced</li>
                    <li>Gluon (g) only with quarks ⭐Enforced</li>
                    <li>W± bosons can change flavor ⭐CKM Matrix</li>
                    <li>Weak interaction violates parity</li>
                    <li>Vertex dimensionality cannot exceed 4.0 ⭐New</li>
                    <li>Coupling constants: αₛ≈0.12, αₑₘ≈1/137, αw≈0.03 ⭐New</li>
                </ul>
            </div>
            <div class="bg-slate-800 p-2 rounded border border-slate-700">
                <div class="font-bold text-slate-300 mb-1 text-xs">💡 Usage Tips</div>
                <ul class="text-slate-400 text-[11px] space-y-0.5 list-disc list-inside">
                    <li>Fermion: L→R = particle, R→L = antiparticle</li>
                    <li>W boson: Select W⁺ or W⁻ in the panel</li>
                    <li>Quarks need color charge selection</li>
                    <li>Gluons need color combination (color-anticolor) ⭐Fixed</li>
                    <li>Blue dashed lines: Initial (left) & Final (right) states ⭐</li>
                    <li>Mass conservation: only checked for external vertices ⭐</li>
                    <li>Internal lines (virtual particles) can violate energy conservation ⭐</li>
                    <li>CKM Matrix: u↔d strongest (97%), u↔b weakest (0.4%) ⭐New</li>
                    <li>Radiative correction: photon/gluon can attach to external lines (e⁻→e⁻+γ) ⭐New</li>
                </ul>
            </div>
        `
    }
};

function t(key, replacements = {}) {
    let text = i18n[currentLang][key] || key;
    for (const placeholder in replacements) {
        text = text.replace(`{${placeholder}}`, replacements[placeholder]);
    }
    return text;
}

function updateLanguage() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.innerHTML = t(key);
    });
    
    document.getElementById('help-content').innerHTML = t('help-content');
    document.getElementById('lang-text').textContent = currentLang === 'zh' ? '中文' : 'EN';
    
    // Re-populate dynamic elements that depend on language
    updateUI();
    populateFermionSelect(currentProps);
    
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}


// --- 物理核心数据库 ---
const PHYSICS = {
    leptons: [
        { id: 'e', name: '电子 (e⁻)', enName: 'Electron (e⁻)', symbol: 'e⁻', charge: -1, lepton: 1, baryon: 0, 
          flavor: 'electron', spin: 0.5, parity: 1, mass: 0.511 },
        { id: 'mu', name: '缪子 (μ⁻)', enName: 'Muon (μ⁻)', symbol: 'μ⁻', charge: -1, lepton: 1, baryon: 0,
          flavor: 'muon', spin: 0.5, parity: 1, mass: 105.7 },
        { id: 'tau', name: '陶子 (τ⁻)', enName: 'Tau (τ⁻)', symbol: 'τ⁻', charge: -1, lepton: 1, baryon: 0,
          flavor: 'tau', spin: 0.5, parity: 1, mass: 1777 },
        { id: 'nu_e', name: '电中微子 (νₑ)', enName: 'Electron neutrino (νₑ)', symbol: 'νₑ', charge: 0, lepton: 1, baryon: 0,
          flavor: 'electron', spin: 0.5, parity: -1, mass: 0.001 },
        { id: 'nu_mu', name: '缪中微子 (ν_μ)', enName: 'Muon neutrino (ν_μ)', symbol: 'ν_μ', charge: 0, lepton: 1, baryon: 0,
          flavor: 'muon', spin: 0.5, parity: -1, mass: 0.001 },
        { id: 'nu_tau', name: '陶中微子 (ν_τ)', enName: 'Tau neutrino (ν_τ)', symbol: 'ν_τ', charge: 0, lepton: 1, baryon: 0,
          flavor: 'tau', spin: 0.5, parity: -1, mass: 0.001 }
    ],
    quarks_u: [
        { id: 'u', name: '上夸克 (u)', enName: 'Up quark (u)', symbol: 'u', charge: 2/3, lepton: 0, baryon: 1/3,
          flavor: 'up', spin: 0.5, parity: 1, mass: 2.2, color: null },
        { id: 'c', name: '魅力 (c)', enName: 'Charm (c)', symbol: 'c', charge: 2/3, lepton: 0, baryon: 1/3,
          flavor: 'charm', spin: 0.5, parity: 1, mass: 1280, color: null },
        { id: 't', name: '顶夸克 (t)', enName: 'Top (t)', symbol: 't', charge: 2/3, lepton: 0, baryon: 1/3,
          flavor: 'top', spin: 0.5, parity: 1, mass: 173000, color: null }
    ],
    quarks_d: [
        { id: 'd', name: '下夸克 (d)', enName: 'Down quark (d)', symbol: 'd', charge: -1/3, lepton: 0, baryon: 1/3,
          flavor: 'down', spin: 0.5, parity: 1, mass: 4.7, color: null },
        { id: 's', name: '奇异 (s)', enName: 'Strange (s)', symbol: 's', charge: -1/3, lepton: 0, baryon: 1/3,
          flavor: 'strange', spin: 0.5, parity: 1, mass: 96, color: null },
        { id: 'b', name: '底夸克 (b)', enName: 'Bottom (b)', symbol: 'b', charge: -1/3, lepton: 0, baryon: 1/3,
          flavor: 'bottom', spin: 0.5, parity: 1, mass: 4180, color: null }
    ],
    bosons: {
        'photon': { id: 'photon', name: '光子 (γ)', enName: 'Photon (γ)', symbol: 'γ', charge: 0, lepton: 0, baryon: 0,
                   spin: 1, parity: -1, mass: 0, colorCharge: null },
        'gluon': { id: 'gluon', name: '胶子 (g)', enName: 'Gluon (g)', symbol: 'g', charge: 0, lepton: 0, baryon: 0,
                  spin: 1, parity: -1, mass: 0, colorCharge: null },
        'w_plus': { id: 'w_plus', name: 'W⁺', enName: 'W⁺', symbol: 'W⁺', charge: 1, lepton: 0, baryon: 0,
                   spin: 1, parity: 1, mass: 80379, flavorChange: true },
        'w_minus': { id: 'w_minus', name: 'W⁻', enName: 'W⁻', symbol: 'W⁻', charge: -1, lepton: 0, baryon: 0,
                    spin: 1, parity: 1, mass: 80379, flavorChange: true },
        'z': { id: 'z', name: 'Z⁰', enName: 'Z⁰', symbol: 'Z⁰', charge: 0, lepton: 0, baryon: 0,
              spin: 1, parity: 1, mass: 91188 },
        'higgs': { id: 'higgs', name: '希格斯 (H)', enName: 'Higgs (H)', symbol: 'H', charge: 0, lepton: 0, baryon: 0,
                  spin: 0, parity: 1, mass: 125100 }
    }
};

const COLORS = ['red', 'green', 'blue', 'anti-red', 'anti-green', 'anti-blue'];

const GLUON_COLORS = [
    { in: 'red', out: 'anti-green', label: 'rḡ', display: 'Red-antiGreen' },
    { in: 'red', out: 'anti-blue', label: 'rb̄', display: 'Red-antiBlue' },
    { in: 'green', out: 'anti-red', label: 'gr̄', display: 'Green-antiRed' },
    { in: 'green', out: 'anti-blue', label: 'gb̄', display: 'Green-antiBlue' },
    { in: 'blue', out: 'anti-red', label: 'br̄', display: 'Blue-antiRed' },
    { in: 'blue', out: 'anti-green', label: 'bḡ', display: 'Blue-antiGreen' }
];

// --- 粒子质量数据库 (MeV) ---
const PARTICLE_MASSES = {
    // 轻子 Leptons
    'e': 0.511,
    'mu': 105.66,
    'tau': 1776.8,
    'nu_e': 0.0,
    'nu_mu': 0.0,
    'nu_tau': 0.0,
    // 夸克 Quarks
    'u': 2.2,
    'd': 4.7,
    'c': 1280,
    's': 96,
    't': 173100,
    'b': 4180,
    // 玻色子 Bosons
    'photon': 0.0,
    'gluon': 0.0,
    'w_plus': 80379,
    'w_minus': 80379,
    'z': 91188,
    'higgs': 125100
};

// 粒子维度常数 (用于防止非物理高阶顶点)
const FERMION_DIMENSIONALITY = 1.5;
const BOSON_DIMENSIONALITY = 1.0;
const MAXIMUM_DIMENSIONALITY = 4.0;

// === CKM Matrix for Quark Mixing (Weak Interactions) ===
const CKM_MATRIX = {
    'u': { 'd': 0.97370, 's': 0.2245, 'b': 0.00382 },
    'c': { 'd': 0.221, 's': 0.987, 'b': 0.041 },
    't': { 'd': 0.008, 's': 0.0388, 'b': 1.013 }
};

// === Coupling Constants (Fine Structure Constants) ===
const ALPHA_EM = 1.0 / 137.0;      // Electromagnetic coupling
const ALPHA_S = 0.1181;             // Strong coupling (QCD)
const ALPHA_W = 1.0 / 30.0;         // Weak coupling

// --- Canvas 引擎 ---
const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const dpr = window.devicePixelRatio || 1;
let shapes = [];
let currentTool = 'select';
let selectedShapeIds = new Set();

let isDrawing = false;
let draftLine = null;
let snapPoint = null;

// 初态和末态边界线位置（距离画布边缘的像素）
const INITIAL_STATE_X = 80;  // 左边界
const FINAL_STATE_X_OFFSET = 80;  // 右边界距离右侧的距离

let currentProps = {
    category: 'fermion',
    group: 'lepton',
    particleId: 'e',
    color: null,
    gluonColor: null
};

// --- UI 元素 ---
const ui = {
    empty: document.getElementById('empty-state'),
    fermion: document.getElementById('settings-fermion'),
    boson: document.getElementById('settings-boson'),
    selParticle: document.getElementById('sel-particle'),
    selColor: document.getElementById('sel-color'),
    colorSelector: document.getElementById('color-selector'),
    selGluonColor: document.getElementById('sel-gluon-color'),
    gluonColorSelector: document.getElementById('gluon-color-selector'),
    previewQ: document.getElementById('preview-q'),
    previewL: document.getElementById('preview-l'),
    previewB: document.getElementById('preview-b'),
    previewSpin: document.getElementById('preview-spin'),
    previewParity: document.getElementById('preview-parity')
};

// --- 初始化 ---
function init() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);
    
    updateLanguage(); // Set initial language
    updateUI();
    draw();
}

window.addEventListener('resize', () => {
    clearTimeout(window.resizeTimeout);
    window.resizeTimeout = setTimeout(init, 100);
});

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(init, 50);
});


// --- 绘图逻辑 ---
function draw() {
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);

    // 绘制初态和末态边界线
    ctx.save();
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    
    // 初态线（左侧）
    ctx.beginPath();
    ctx.moveTo(INITIAL_STATE_X, 0);
    ctx.lineTo(INITIAL_STATE_X, h);
    ctx.stroke();
    
    // 末态线（右侧）
    const finalStateX = w - FINAL_STATE_X_OFFSET;
    ctx.beginPath();
    ctx.moveTo(finalStateX, 0);
    ctx.lineTo(finalStateX, h);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // 添加标签
    ctx.fillStyle = '#3b82f6';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(currentLang === 'zh' ? '初态' : 'Initial', INITIAL_STATE_X, 20);
    ctx.fillText(currentLang === 'zh' ? '末态' : 'Final', finalStateX, 20);
    
    ctx.restore();

    shapes.forEach(s => drawShape(s));

    if (draftLine) {
        const tempShape = {
            ...draftLine,
            type: currentTool,
            props: { ...currentProps },
            id: 'draft'
        };
        
        if (tempShape.type === 'fermion') {
            tempShape.props.isAnti = (tempShape.p2.x < tempShape.p1.x);
        }
        
        if (currentTool === 'boson_w') {
            if (tempShape.p2.x < tempShape.p1.x) {
                tempShape.props.particleId = 'w_plus';
            } else {
                tempShape.props.particleId = 'w_minus';
            }
        }
        
        ctx.globalAlpha = 0.6;
        drawShape(tempShape);
        ctx.globalAlpha = 1.0;
    }
    
    // 自动保存到画布管理器（节流：仅在没有草稿时保存）
    if (!draftLine && !window._disableAutoSave) {
        saveToCanvasManager();
    }
}

// 节流保存：避免频繁保存
let saveTimeout = null;
function saveToCanvasManager() {
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
        const canvasManager = window.getCanvasManager();
        if (canvasManager && !window._disableAutoSave) {
            canvasManager.updateCurrentCanvas(shapes);
        }
    }, 500); // 500ms 后保存
}

function drawShape(s) {
    const isSelected = selectedShapeIds.has(s.id);
    
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    let color = '#94a3b8';
    if (s.type === 'fermion') color = s.props.isAnti ? '#f87171' : '#34d399';
    if (s.type === 'photon') color = '#facc15';
    if (s.type === 'boson_w') color = '#fb923c';
    if (s.type === 'boson_z') color = '#60a5fa';
    if (s.type === 'gluon') color = '#ec4899';
    if (s.type === 'higgs') color = '#a855f7';
    if (isSelected) color = '#22d3ee';

    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    const dx = s.p2.x - s.p1.x;
    const dy = s.p2.y - s.p1.y;
    const dist = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx);

    ctx.save();
    ctx.translate(s.p1.x, s.p1.y);
    ctx.rotate(angle);

    if (s.type === 'fermion') {
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(dist, 0); ctx.stroke();
        
        const mid = dist / 2;
        ctx.translate(mid, 0);

        const arrowSize = 8;
        ctx.beginPath();
        ctx.moveTo(arrowSize * 0.5, 0);
        ctx.lineTo(-arrowSize * 0.5, -arrowSize/1.6);
        ctx.lineTo(-arrowSize * 0.5, arrowSize/1.6);
        ctx.closePath();
        ctx.fill();

    } else if (s.type === 'photon' || s.type === 'boson_w' || s.type === 'boson_z') {
        ctx.beginPath();
        ctx.moveTo(0,0);
        const freq = s.type === 'photon' ? 0.2 : 0.3;
        const amp = s.type === 'photon' ? 5 : 4;
        for(let i=0; i<=dist; i+=2) {
            ctx.lineTo(i, Math.sin(i*freq)*amp);
        }
        ctx.stroke();
    } else if (s.type === 'gluon') {
        ctx.beginPath();
        const r = 4;
        for(let i=0; i<=dist; i++) {
            const t = i * 0.4;
            const cx = i + Math.cos(t+Math.PI)*r*0.8;
            const cy = Math.sin(t+Math.PI)*r;
            if(i===0) ctx.moveTo(cx,cy); else ctx.lineTo(cx,cy);
        }
        ctx.stroke();
    } else if (s.type === 'higgs') {
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(dist, 0); ctx.stroke();
        ctx.setLineDash([]);
    }

    ctx.restore();

    let label = getParticleSymbol(s.props);
    if (s.props.isAnti && s.type === 'fermion') label = convertToAntiSymbol(label);
    
    if (s.props.color && (s.type === 'fermion')) {
        const colorLabels = {
            'red': 'ᴿ', 'green': 'ᴳ', 'blue': 'ᴮ',
            'anti-red': 'R̄', 'anti-green': 'Ḡ', 'anti-blue': 'B̄'
        };
        label += colorLabels[s.props.color] || '';
    }
    
    if (s.type === 'gluon' && s.props.gluonColor !== null && s.props.gluonColor !== undefined) {
        const gc = GLUON_COLORS[s.props.gluonColor];
        if (gc) label += `(${gc.label})`;
    }
    
    const midX = (s.p1.x + s.p2.x)/2;
    const midY = (s.p1.y + s.p2.y)/2;
    
    ctx.fillStyle = '#fff';
    ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(label, midX, midY - 10);
    
    if(isSelected) {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 1;
        ctx.strokeRect(s.p1.x-4, s.p1.y-4, 8, 8);
        ctx.strokeRect(s.p2.x-4, s.p2.y-4, 8, 8);
    }
}

// --- 交互逻辑 ---
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return {x, y};
}

function checkSnap(pos) {
    const threshold = 15;
    const w = canvas.width / dpr;
    const finalStateX = w - FINAL_STATE_X_OFFSET;
    
    // 检查是否靠近初态边界线
    if (Math.abs(pos.x - INITIAL_STATE_X) < threshold) {
        return { x: INITIAL_STATE_X, y: pos.y };
    }
    
    // 检查是否靠近末态边界线
    if (Math.abs(pos.x - finalStateX) < threshold) {
        return { x: finalStateX, y: pos.y };
    }
    
    // 检查现有粒子的端点
    for(let s of shapes) {
        if (Math.hypot(s.p1.x - pos.x, s.p1.y - pos.y) < threshold) return s.p1;
        if (Math.hypot(s.p2.x - pos.x, s.p2.y - pos.y) < threshold) return s.p2;
    }
    
    // ⭐ 新增：检查是否靠近线段中间（用于辐射修正）
    // 计算鼠标位置到每条线段的最近点
    for(let s of shapes) {
        const A = pos.x - s.p1.x;
        const B = pos.y - s.p1.y;
        const C = s.p2.x - s.p1.x;
        const D = s.p2.y - s.p1.y;
        const dot = A*C + B*D;
        const len_sq = C*C + D*D;
        
        if (len_sq === 0) continue; // 退化的点
        
        let param = dot / len_sq;
        
        // 只有当点在线段中间时才吸附（不在端点附近）
        if (param > 0.1 && param < 0.9) {
            const nearestX = s.p1.x + param * C;
            const nearestY = s.p1.y + param * D;
            const dist = Math.hypot(pos.x - nearestX, pos.y - nearestY);
            
            if (dist < threshold) {
                return { x: nearestX, y: nearestY };
            }
        }
    }
    
    return null;
}

canvas.addEventListener('pointerdown', e => {
    const pos = getPos(e);
    const snap = checkSnap(pos);
    const startPoint = snap || pos;

    if (currentTool === 'select') {
        let hit = null;
        for (let i = shapes.length - 1; i >= 0; i--) {
            const s = shapes[i];
            const A = pos.x - s.p1.x; const B = pos.y - s.p1.y;
            const C = s.p2.x - s.p1.x; const D = s.p2.y - s.p1.y;
            const dot = A*C + B*D;
            const len_sq = C*C + D*D;
            let param = -1;
            if (len_sq != 0) param = dot / len_sq;
            let xx, yy;
            if (param < 0) { xx = s.p1.x; yy = s.p1.y; }
            else if (param > 1) { xx = s.p2.x; yy = s.p2.y; }
            else { xx = s.p1.x + param*C; yy = s.p1.y + param*D; }
            const dist = Math.hypot(pos.x - xx, pos.y - yy);
            if(dist < 10) {
                hit = s;
                break;
            }
        }

        if (hit) {
            if (selectedShapeIds.has(hit.id)) {
                selectedShapeIds.delete(hit.id);
            } else {
                selectedShapeIds.add(hit.id);
            }
        } else {
            selectedShapeIds.clear();
        }
        draw();
        updateUI();
    } else {
        isDrawing = true;
        draftLine = { p1: startPoint, p2: pos };
        selectedShapeIds.clear();
    }
});

canvas.addEventListener('pointermove', e => {
    const pos = getPos(e);
    const snap = checkSnap(pos);
    
    const indicator = document.getElementById('snap-indicator');
    if (snap) {
        indicator.style.display = 'block';
        indicator.style.left = snap.x + 'px';
        indicator.style.top = snap.y + 'px';
        snapPoint = snap;
    } else {
        indicator.style.display = 'none';
        snapPoint = null;
    }

    if (isDrawing) {
        draftLine.p2 = snap || pos;
        draw();
    }
});

canvas.addEventListener('pointerup', e => {
    if (isDrawing && draftLine) {
        const dist = Math.hypot(draftLine.p1.x - draftLine.p2.x, draftLine.p1.y - draftLine.p2.y);
        if (dist > 10) {
            let finalProps = JSON.parse(JSON.stringify(currentProps));
            
            if (currentTool === 'fermion') {
                finalProps.isAnti = (draftLine.p2.x < draftLine.p1.x);
            } else if (currentTool === 'boson_w') {
                if (draftLine.p2.x < draftLine.p1.x) {
                    finalProps.particleId = 'w_plus';
                } else {
                    finalProps.particleId = 'w_minus';
                }
            }

            shapes.push({
                id: Date.now(),
                type: currentTool,
                p1: draftLine.p1,
                p2: draftLine.p2,
                props: finalProps
            });
        }
    }
    isDrawing = false;
    draftLine = null;
    draw();
});

// --- 物理验证系统 ---
document.getElementById('btn-validate').onclick = validatePhysics;

function validatePhysics() {
    document.getElementById('error-layer').innerHTML = '';
    
    let vertices = [];
    const SNAP_DIST = 15;
    const w = canvas.width / dpr;
    const finalStateX = w - FINAL_STATE_X_OFFSET;

    // 辅助函数：判断点是否在初态或末态边界上（外线）
    function isOnBoundary(x) {
        return Math.abs(x - INITIAL_STATE_X) < SNAP_DIST || 
               Math.abs(x - finalStateX) < SNAP_DIST;
    }

    function getVertex(p) {
        let v = vertices.find(v => Math.hypot(v.x - p.x, v.y - p.y) < SNAP_DIST);
        if (!v) {
            v = { x: p.x, y: p.y, incoming: [], outgoing: [], isExternal: isOnBoundary(p.x) };
            vertices.push(v);
        }
        return v;
    }
    
    // ⭐ 新增：检查点是否在某条线段上（用于检测中间顶点）
    function findPointOnLine(p) {
        for(let s of shapes) {
            // 跳过以该点为端点的线（已经通过getVertex处理）
            if (Math.hypot(s.p1.x - p.x, s.p1.y - p.y) < SNAP_DIST) continue;
            if (Math.hypot(s.p2.x - p.x, s.p2.y - p.y) < SNAP_DIST) continue;
            
            // 检查点是否在线段上
            const A = p.x - s.p1.x;
            const B = p.y - s.p1.y;
            const C = s.p2.x - s.p1.x;
            const D = s.p2.y - s.p1.y;
            const dot = A*C + B*D;
            const len_sq = C*C + D*D;
            
            if (len_sq === 0) continue;
            
            const param = dot / len_sq;
            
            // 点在线段内部（不在端点）
            if (param > 0.05 && param < 0.95) {
                const nearestX = s.p1.x + param * C;
                const nearestY = s.p1.y + param * D;
                const dist = Math.hypot(p.x - nearestX, p.y - nearestY);
                
                if (dist < SNAP_DIST) {
                    return { shape: s, point: { x: nearestX, y: nearestY }, param: param };
                }
            }
        }
        return null;
    }

    shapes.forEach(s => {
        const v1 = getVertex(s.p1);
        const v2 = getVertex(s.p2);

        let particleData = getParticleData(s.props);
        if (!particleData) return;

        let flowFromV1ToV2 = true;
        if (s.type === 'fermion' && s.props.isAnti) {
            flowFromV1ToV2 = false;
        }
        // ⭐ 修复：W⁺玻色子的电荷流向与箭头相反
        if (s.type === 'boson_w' && s.props.particleId === 'w_plus') {
            flowFromV1ToV2 = false;
        }

        let extendedData = {
            ...particleData,
            ...s.props,
            color: s.props.color || null,
            gluonColor: s.props.gluonColor !== undefined ? s.props.gluonColor : null,
            isAnti: s.props.isAnti || false,
            shapeId: s.id,
            type: s.type
        };

        if (s.type === 'fermion' && s.props.isAnti) {
            extendedData.charge = -particleData.charge;
            extendedData.lepton = -particleData.lepton;
            extendedData.baryon = -particleData.baryon;
        }

        if (flowFromV1ToV2) {
            v1.outgoing.push(extendedData);
            v2.incoming.push(extendedData);
        } else {
            v2.outgoing.push(extendedData);
            v1.incoming.push(extendedData);
        }
    });
    
    // ⭐ 处理线段中间的顶点（辐射修正）
    // 检查每条线的端点是否落在其他线段上
    shapes.forEach(line => {
        // 检查这条线的两个端点
        [line.p1, line.p2].forEach(endpoint => {
            const onLineResult = findPointOnLine(endpoint);
            if (onLineResult) {
                // 找到了！这个端点在另一条线段上
                const { shape: baseLine, point: intersectionPoint } = onLineResult;
                
                // 创建或获取交点顶点
                const intersectionVertex = getVertex(intersectionPoint);
                
                // 将基线（被穿过的线）在这个点分成两段
                // 基线的粒子数据
                let baseParticleData = getParticleData(baseLine.props);
                if (!baseParticleData) return;
                
                let baseFlowFromV1ToV2 = true;
                if (baseLine.type === 'fermion' && baseLine.props.isAnti) {
                    baseFlowFromV1ToV2 = false;
                }
                // ⭐ 修复：W⁺玻色子的电荷流向与箭头相反
                if (baseLine.type === 'boson_w' && baseLine.props.particleId === 'w_plus') {
                    baseFlowFromV1ToV2 = false;
                }
                
                let baseExtendedData = {
                    ...baseParticleData,
                    ...baseLine.props,
                    color: baseLine.props.color || null,
                    gluonColor: baseLine.props.gluonColor !== undefined ? baseLine.props.gluonColor : null,
                    isAnti: baseLine.props.isAnti || false,
                    shapeId: baseLine.id,
                    type: baseLine.type
                };
                
                if (baseLine.type === 'fermion' && baseLine.props.isAnti) {
                    baseExtendedData.charge = -baseParticleData.charge;
                    baseExtendedData.lepton = -baseParticleData.lepton;
                    baseExtendedData.baryon = -baseParticleData.baryon;
                }
                
                // 基线流经这个交点
                // 将基线视为 incoming 到交点，然后 outgoing 从交点
                if (baseFlowFromV1ToV2) {
                    intersectionVertex.incoming.push(baseExtendedData);
                    intersectionVertex.outgoing.push(baseExtendedData);
                } else {
                    intersectionVertex.outgoing.push(baseExtendedData);
                    intersectionVertex.incoming.push(baseExtendedData);
                }
            }
        });
    });

    let allValid = true;
    let vertexCount = 0;
    let errorCount = 0;
    let warnings = [];

    vertices.forEach(v => {
        const totalLines = v.incoming.length + v.outgoing.length;
        
        // 跳过孤立点（少于2条线）
        if (totalLines < 2) return;
        
        // 检测特殊情况：2-粒子顶点（如 e⁻ + γ → e⁻）
        // 这种情况下，一个粒子可能既是incoming又是outgoing（外线上的辐射）
        const is2ParticleVertex = totalLines === 2;
        
        vertexCount++;
        
        let sumQ = 0, sumL = 0, sumB = 0;
        let sumParity = 1;
        let sumDimensionality = 0;
        let massIn = 0, massOut = 0;
        let colorIn = {}, colorOut = {};
        let flavorsIn = new Set(), flavorsOut = new Set();
        
        let hasPhoton = false, hasGluon = false, hasZ = false;
        let hasChargedParticle = false, hasQuark = false;
        let hasFermion = false;
        
        v.incoming.forEach(p => {
            sumQ += p.charge;
            sumL += p.lepton;
            sumB += p.baryon;
            sumParity *= p.parity || 1;
            sumDimensionality += getParticleDimensionality(p);
            massIn += getParticleMass(p);
            
            if (p.color) colorIn[p.color] = (colorIn[p.color] || 0) + 1;
            if (p.gluonColor !== null && p.gluonColor !== undefined) {
                const gc = GLUON_COLORS[p.gluonColor];
                if (gc) {
                    colorIn[gc.in] = (colorIn[gc.in] || 0) + 1;
                    colorOut[gc.out] = (colorOut[gc.out] || 0) + 1;
                }
            }
            if (p.flavor) flavorsIn.add(p.isAnti ? 'anti-' + p.flavor : p.flavor);
            
            // 检测粒子类型
            if (p.type === 'photon') hasPhoton = true;
            if (p.type === 'gluon') hasGluon = true;
            if (p.type === 'boson_z') hasZ = true;
            if (hasCharge(p)) hasChargedParticle = true;
            if (isQuark(p)) hasQuark = true;
            if (p.type === 'fermion') hasFermion = true;
        });
        
        v.outgoing.forEach(p => {
            sumQ -= p.charge;
            sumL -= p.lepton;
            sumB -= p.baryon;
            sumParity *= p.parity || 1;
            sumDimensionality += getParticleDimensionality(p);
            massOut += getParticleMass(p);
            
            if (p.color) colorOut[p.color] = (colorOut[p.color] || 0) + 1;
            if (p.gluonColor !== null && p.gluonColor !== undefined) {
                const gc = GLUON_COLORS[p.gluonColor];
                if (gc) {
                    colorOut[gc.in] = (colorOut[gc.in] || 0) + 1;
                    colorIn[gc.out] = (colorIn[gc.out] || 0) + 1;
                }
            }
            if (p.flavor) flavorsOut.add(p.isAnti ? 'anti-' + p.flavor : p.flavor);
            
            // 检测粒子类型
            if (p.type === 'photon') hasPhoton = true;
            if (p.type === 'gluon') hasGluon = true;
            if (p.type === 'boson_z') hasZ = true;
            if (hasCharge(p)) hasChargedParticle = true;
            if (isQuark(p)) hasQuark = true;
            if (p.type === 'fermion') hasFermion = true;
        });

        let conservationViolated = false;
        let msg = [];
        
        // 特殊处理：2-粒子顶点（外线辐射）
        // 例如：e⁻(initial) + γ → e⁻(final)
        // 这在物理上是允许的（初态/末态辐射）
        if (is2ParticleVertex) {
            // 检查是否是有效的2-粒子顶点
            // 允许的情况：
            // 1. 费米子 + 光子（QED辐射）
            // 2. 夸克 + 胶子（QCD辐射）
            // 3. 费米子 + Z/W（弱相互作用）
            
            const hasValidRadiativeVertex = (
                (hasFermion && hasPhoton && hasChargedParticle) ||  // e + γ
                (hasQuark && hasGluon) ||                            // q + g
                (hasFermion && (v.incoming.some(p => p.type === 'boson_z' || p.type === 'boson_w') || 
                                v.outgoing.some(p => p.type === 'boson_z' || p.type === 'boson_w')))
            );
            
            if (!hasValidRadiativeVertex) {
                msg.push(`2-粒子顶点无效（不是有效的辐射过程）`);
                conservationViolated = true;
            }
            
            // 对于2-粒子顶点，放宽一些守恒律检查
            // 因为incoming和outgoing的定义可能不明确
            // 但仍然要检查基本的粒子类型匹配
        }
        
        // 1. 电荷守恒（2-粒子顶点可能需要特殊处理）
        if (!is2ParticleVertex || (is2ParticleVertex && Math.abs(sumQ) > 0.01)) {
            if (Math.abs(sumQ) > 0.01) { 
                msg.push(`电荷不守恒 (ΔQ=${sumQ.toFixed(2)})`); 
                conservationViolated = true; 
            }
        }
        
        // 2. 轻子数守恒
        if (!is2ParticleVertex || (is2ParticleVertex && Math.abs(sumL) > 0.01)) {
            if (Math.abs(sumL) > 0.01) { 
                msg.push(`轻子数不守恒 (ΔL=${sumL.toFixed(2)})`); 
                conservationViolated = true; 
            }
        }
        
        // 3. 重子数守恒
        if (!is2ParticleVertex || (is2ParticleVertex && Math.abs(sumB) > 0.01)) {
            if (Math.abs(sumB) > 0.01) { 
                msg.push(`重子数不守恒 (ΔB=${sumB.toFixed(2)})`); 
                conservationViolated = true; 
            }
        }
        
        // 4. 能量-动量守恒（质量检查）
        // 物理规则：
        // - 对于外部顶点（在边界线上）：输入和输出必须分别在边界的两侧
        // - 对于内部顶点（不在边界线上）：即使连接外线，也可以违反能量守恒（虚粒子传播）
        const MASS_TOLERANCE = 0.1; // MeV
        
        // 只对位于边界线上的顶点进行严格的能量守恒检查
        // 内部顶点即使连接外线也允许违反能量守恒
        if (v.isExternal && massOut > massIn + MASS_TOLERANCE) {
            msg.push(`能量守恒违反 (质量: ${massIn.toFixed(1)}→${massOut.toFixed(1)} MeV)`);
            conservationViolated = true;
        }
        
        // 5. 维度检查（防止非物理高阶顶点）
        if (sumDimensionality > MAXIMUM_DIMENSIONALITY) {
            msg.push(`顶点维度过高 (D=${sumDimensionality.toFixed(1)} > ${MAXIMUM_DIMENSIONALITY})`);
            conservationViolated = true;
        }
        
        // 6. 禁止规则：光子必须与带电粒子耦合
        if (hasPhoton && !hasChargedParticle) {
            msg.push(`光子必须与带电粒子相互作用`);
            conservationViolated = true;
        }
        
        // 7. 胶子耦合规则：胶子可以与夸克耦合，也可以自耦合（3胶子顶点、4胶子顶点）
        // ✅ 移除了错误的"胶子只能与夸克耦合"规则
        // QCD 允许胶子自相互作用！
        
        // 8. 色荷守恒 (修复版 - 正确处理反色荷)
        if (hasGluon || hasQuark) {
            let colorViolation = false;
            let colorBalance = { 'red': 0, 'green': 0, 'blue': 0 };
            
            // 计算净色荷流入
            v.incoming.forEach(p => {
                if (p.color) {
                    // ✅ 修复：正确处理反色荷
                    if (p.color.startsWith('anti-')) {
                        // 反色荷流入 = 正色荷流出（负贡献）
                        const baseColor = p.color.replace('anti-', '');
                        colorBalance[baseColor] = (colorBalance[baseColor] || 0) - 1;
                    } else {
                        // 正色荷流入
                        colorBalance[p.color] = (colorBalance[p.color] || 0) + 1;
                    }
                }
                if (p.gluonColor !== null && p.gluonColor !== undefined) {
                    const gc = GLUON_COLORS[p.gluonColor];
                    if (gc) {
                        // 胶子流入：带来正色荷 gc.in 和反色荷 gc.out
                        colorBalance[gc.in] = (colorBalance[gc.in] || 0) + 1;
                        const baseColor = gc.out.replace('anti-', '');
                        colorBalance[baseColor] = (colorBalance[baseColor] || 0) - 1;
                    }
                }
            });
            
            // 计算净色荷流出
            v.outgoing.forEach(p => {
                if (p.color) {
                    // ✅ 修复：正确处理反色荷
                    if (p.color.startsWith('anti-')) {
                        // 反色荷流出 = 正色荷流入（正贡献）
                        const baseColor = p.color.replace('anti-', '');
                        colorBalance[baseColor] = (colorBalance[baseColor] || 0) + 1;
                    } else {
                        // 正色荷流出
                        colorBalance[p.color] = (colorBalance[p.color] || 0) - 1;
                    }
                }
                if (p.gluonColor !== null && p.gluonColor !== undefined) {
                    const gc = GLUON_COLORS[p.gluonColor];
                    if (gc) {
                        // 胶子流出：带走正色荷 gc.in 和反色荷 gc.out
                        colorBalance[gc.in] = (colorBalance[gc.in] || 0) - 1;
                        const baseColor = gc.out.replace('anti-', '');
                        colorBalance[baseColor] = (colorBalance[baseColor] || 0) + 1;
                    }
                }
            });
            
            // 检查所有色荷是否平衡
            for (let color in colorBalance) {
                if (Math.abs(colorBalance[color]) > 0.01) {
                    colorViolation = true;
                    console.log(`❌ 色荷不守恒 at 顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}):`, colorBalance);
                    break;
                }
            }
            
            if (colorViolation) { 
                msg.push(`色荷不守恒`); 
                warnings.push(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}): 色荷必须形成色单态（总色荷=白色）`); 
            }
        }
        
        // 9. 宇称守恒（弱相互作用可破坏）
        const hasWeakBoson = v.incoming.some(p => p.flavorChange) || v.outgoing.some(p => p.flavorChange);
        if (!hasWeakBoson && sumParity !== 1 && sumParity !== -1) { 
            warnings.push(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}): 宇称可能不守恒 (Π=${sumParity.toFixed(2)})`); 
        }
        
        // 10. 味守恒（弱相互作用可破坏）+ CKM 矩阵检查
        if (hasWeakBoson) {
            // 检查夸克味变换是否符合CKM矩阵允许的跃迁
            let quarkTransitions = [];
            v.incoming.forEach(p => {
                if (isQuark(p)) {
                    const outQuark = v.outgoing.find(q => isQuark(q) && q.particleId !== p.particleId);
                    if (outQuark) {
                        quarkTransitions.push({from: p.particleId, to: outQuark.particleId});
                    }
                }
            });
            
            quarkTransitions.forEach(trans => {
                if (CKM_MATRIX[trans.from] && CKM_MATRIX[trans.from][trans.to]) {
                    const prob = CKM_MATRIX[trans.from][trans.to];
                    if (prob < 0.01) {
                        warnings.push(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}): 夸克跃迁 ${trans.from}→${trans.to} 的CKM概率极低 (${(prob*100).toFixed(2)}%)`);
                    }
                }
            });
        } else {
            const inFlavors = Array.from(flavorsIn).sort().join(',');
            const outFlavors = Array.from(flavorsOut).sort().join(',');
            if (inFlavors !== outFlavors && flavorsIn.size > 0 && flavorsOut.size > 0) { 
                warnings.push(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}): 味发生改变但无弱玻色子`); 
            }
        }
        
        if (conservationViolated) {
            allValid = false;
            errorCount++;
            showError(v.x, v.y, msg.join('<br>'));
        }
    });
    
    // ⭐ 显示所有警告（黄色）
    vertices.forEach(v => {
        const vertexWarnings = warnings.filter(w => 
            w.includes(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)})`)
        );
        if (vertexWarnings.length > 0) {
            const warningMsg = vertexWarnings.map(w => 
                w.replace(`顶点(${v.x.toFixed(0)},${v.y.toFixed(0)}): `, '')
            ).join('<br>');
            showWarning(v.x, v.y, warningMsg);
        }
    });

    if (vertexCount === 0 && shapes.length > 0) { 
        showToast(t('toast-no-vertex'), "info"); 
    }
    else if (vertexCount > 0 && allValid) {
        let msg = t('toast-validate-pass', {count: vertexCount});
        if (warnings.length > 0) { 
            msg += `\n` + t('toast-validate-warn', {count: warnings.length}); 
            console.log('=== 物理规则警告 ==='); 
            warnings.forEach(w => console.log('⚠️', w)); 
        }
        showToast(msg, warnings.length > 0 ? "warning" : "success");
    } else if (!allValid) { 
        showToast(t('toast-validate-error', {count: errorCount}), "error"); 
    }
    else { 
        showToast(t('toast-empty'), "info"); 
    }
    
    if (warnings.length > 0) { 
        console.log('=== 物理规则警告 ==='); 
        warnings.forEach(w => console.log('⚠️', w)); 
    }
}

function showError(x, y, html) {
    const div = document.createElement('div');
    div.className = 'absolute bg-red-500/90 text-white text-xs px-2 py-1 rounded border border-red-300 shadow-xl z-50 text-center pointer-events-none whitespace-nowrap';
    div.style.left = x + 'px';
    div.style.top = (y + 15) + 'px';
    div.style.transform = 'translateX(-50%)';
    div.innerHTML = `<i data-lucide="alert-circle" class="w-3 h-3 inline mr-1"></i>` + html;
    document.getElementById('error-layer').appendChild(div);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    
    const ring = document.createElement('div');
    ring.className = 'absolute w-8 h-8 border-2 border-red-500 rounded-full -translate-x-1/2 -translate-y-1/2 snap-ring pointer-events-none';
    ring.style.left = x + 'px';
    ring.style.top = y + 'px';
    document.getElementById('error-layer').appendChild(ring);

    setTimeout(() => {
        if (div) div.remove();
        if (ring) ring.remove();
    }, 3000);
}

function showWarning(x, y, html) {
    const div = document.createElement('div');
    div.className = 'absolute bg-yellow-500/90 text-black text-xs px-2 py-1 rounded border border-yellow-300 shadow-xl z-50 text-center pointer-events-none whitespace-nowrap';
    div.style.left = x + 'px';
    div.style.top = (y - 25) + 'px';  // 显示在顶点上方，避免与错误重叠
    div.style.transform = 'translateX(-50%)';
    div.innerHTML = `<i data-lucide="alert-triangle" class="w-3 h-3 inline mr-1"></i>` + html;
    document.getElementById('error-layer').appendChild(div);
    if (typeof lucide !== 'undefined') lucide.createIcons();
    
    const ring = document.createElement('div');
    ring.className = 'absolute w-8 h-8 border-2 border-yellow-500 rounded-full -translate-x-1/2 -translate-y-1/2 snap-ring pointer-events-none';
    ring.style.left = x + 'px';
    ring.style.top = y + 'px';
    document.getElementById('error-layer').appendChild(ring);

    setTimeout(() => {
        if (div) div.remove();
        if (ring) ring.remove();
    }, 4000);  // 警告显示时间稍长一点
}

// --- 辅助函数 ---
function getParticleData(props) {
    if (!props) return null;
    if (props.category === 'fermion') {
        let list = [];
        if(props.group === 'lepton') list = PHYSICS.leptons;
        else if(props.group === 'quark_u') list = PHYSICS.quarks_u;
        else if(props.group === 'quark_d') list = PHYSICS.quarks_d;
        return list.find(p => p.id === props.particleId);
    } else {
        return PHYSICS.bosons[props.particleId];
    }
}

function getParticleMass(props) {
    if (!props) return 0;
    
    // 获取基础粒子ID
    let particleId = props.particleId;
    
    // 处理W玻色子方向
    if (props.category === 'boson') {
        if (particleId === 'w_plus' || particleId === 'w_minus') {
            return PARTICLE_MASSES['w_plus']; // W+和W-质量相同
        }
        if (particleId === 'z') return PARTICLE_MASSES['z'];
        if (particleId === 'higgs') return PARTICLE_MASSES['higgs'];
        if (particleId === 'photon') return PARTICLE_MASSES['photon'];
        if (particleId === 'gluon') return PARTICLE_MASSES['gluon'];
    }
    
    // 费米子
    return PARTICLE_MASSES[particleId] || 0;
}

function getParticleDimensionality(props) {
    if (!props) return 0;
    
    // 玻色子维度为1.0，费米子维度为1.5
    if (props.category === 'boson') {
        return BOSON_DIMENSIONALITY;
    } else {
        return FERMION_DIMENSIONALITY;
    }
}

function hasCharge(props) {
    const p = getParticleData(props);
    if (!p) return false;
    return Math.abs(p.charge) > 0.01;
}

function isQuark(props) {
    if (!props || props.category !== 'fermion') return false;
    return props.group === 'quark_u' || props.group === 'quark_d';
}

function getVertexInteractionStrength(v) {
    // 计算顶点的相互作用强度
    let hasPhoton = v.incoming.some(p => p.type === 'photon') || v.outgoing.some(p => p.type === 'photon');
    let hasGluon = v.incoming.some(p => p.type === 'gluon') || v.outgoing.some(p => p.type === 'gluon');
    let hasWeakBoson = v.incoming.some(p => p.type === 'boson_w' || p.type === 'boson_z') || 
                       v.outgoing.some(p => p.type === 'boson_w' || p.type === 'boson_z');
    
    if (hasGluon) {
        return { strength: ALPHA_S, force: 'strong', label: 'αₛ ≈ 0.12' };
    }
    if (hasPhoton) {
        // 电磁相互作用强度与电荷成正比
        let charge = 0;
        [...v.incoming, ...v.outgoing].forEach(p => {
            const pd = getParticleData(p);
            if (pd) charge += Math.abs(pd.charge);
        });
        return { strength: ALPHA_EM * charge, force: 'EM', label: `αₑₘ ≈ ${(ALPHA_EM * charge).toExponential(2)}` };
    }
    if (hasWeakBoson) {
        return { strength: ALPHA_W, force: 'weak', label: 'αw ≈ 0.03' };
    }
    
    return { strength: 1.0, force: 'unknown', label: '' };
}

function getParticleSymbol(props) {
    const p = getParticleData(props);
    return p ? p.symbol : '?';
}

function convertToAntiSymbol(symbol) {
    if (symbol.includes('⁻')) return symbol.replace('⁻', '⁺');
    if (symbol.includes('⁺')) return symbol.replace('⁺', '⁻');
    if (symbol.includes('ν')) return symbol.replace('ν', 'ν\u0304');
    const quarks = ['u', 'd', 'c', 's', 't', 'b'];
    if (quarks.includes(symbol)) return symbol + '\u0304';
    return 'anti-' + symbol;
}

// --- UI逻辑 ---
function updateUI() {
    ui.empty.style.display = 'none';
    ui.fermion.style.display = 'none';
    ui.boson.style.display = 'none';

    if (currentTool === 'select') {
        if (selectedShapeIds.size === 1) {
            const s = shapes.find(x => selectedShapeIds.has(x.id));
            if (s) {
                showSettingsForType(s.type, s.props);
            }
        } else if (selectedShapeIds.size > 1) {
            ui.empty.style.display = 'flex';
            ui.empty.innerHTML = t('selected-multi', {count: selectedShapeIds.size});
        } else {
            ui.empty.style.display = 'flex';
            ui.empty.innerHTML = t('empty-select');
        }
    } else {
        showSettingsForType(currentTool, currentProps);
    }
}

function showSettingsForType(type, props) {
    if (type === 'fermion') {
        ui.fermion.style.display = 'grid';
        populateFermionSelect(props);
        updatePhysicsPreview(props);
    } else if (['boson_w', 'boson_z', 'photon', 'gluon', 'higgs'].includes(type)) { 
        ui.boson.style.display = 'block';
        populateBosonSelect(type, props);
    }
}

document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTool = btn.dataset.tool;
        
        if (currentTool === 'fermion') currentProps = { category: 'fermion', group: 'lepton', particleId: 'e', color: null };
        if (currentTool === 'boson_w') currentProps = { category: 'boson', particleId: 'w_minus' };
        if (currentTool === 'boson_z') currentProps = { category: 'boson', particleId: 'z' };
        if (currentTool === 'photon') currentProps = { category: 'boson', particleId: 'photon' };
        if (currentTool === 'gluon') currentProps = { category: 'boson', particleId: 'gluon', gluonColor: 0 };
        if (currentTool === 'higgs') currentProps = { category: 'boson', particleId: 'higgs' };
        
        selectedShapeIds.clear();
        updateUI();
    };
});

document.querySelectorAll('.fam-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.fam-btn').forEach(b => b.classList.remove('active', 'bg-cyan-900', 'text-cyan-300', 'border-cyan-500'));
        btn.classList.add('active', 'bg-cyan-900', 'text-cyan-300', 'border-cyan-500');
        currentProps.group = btn.dataset.fam;
        if(currentProps.group === 'lepton') currentProps.particleId = 'e';
        if(currentProps.group === 'quark_u') currentProps.particleId = 'u';
        if(currentProps.group === 'quark_d') currentProps.particleId = 'd';
        populateFermionSelect(currentProps);
        syncChanges();
    }
});

ui.selParticle.onchange = (e) => { currentProps.particleId = e.target.value; syncChanges(); };
ui.selColor.onchange = (e) => { currentProps.color = e.target.value; syncChanges(); };
ui.selGluonColor.onchange = (e) => { currentProps.gluonColor = parseInt(e.target.value); syncChanges(); };

function populateFermionSelect(props) {
    if (!props) return;
    let list = [];
    const group = props.group || 'lepton';
    if(group === 'lepton') list = PHYSICS.leptons;
    else if(group === 'quark_u') list = PHYSICS.quarks_u;
    else if(group === 'quark_d') list = PHYSICS.quarks_d;
    
    ui.selParticle.innerHTML = list.map(p => {
        const name = currentLang === 'zh' ? p.name : p.enName;
        return `<option value="${p.id}" ${p.id===props.particleId?'selected':''}>${name}</option>`;
    }).join('');
    
    document.querySelectorAll('.fam-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.fam === group);
    });
    
    if (group === 'quark_u' || group === 'quark_d') {
        ui.colorSelector.classList.remove('hidden');
        if (!props.color) { currentProps.color = 'red'; ui.selColor.value = 'red'; }
    } else {
        ui.colorSelector.classList.add('hidden');
        currentProps.color = null;
    }
}

function populateBosonSelect(toolType, props) {
    if (!props) return;
    const container = document.getElementById('boson-type-container');
    
    if (toolType === 'boson_w') {
        container.innerHTML = `<div class="text-slate-300 text-sm"><span class="font-mono text-orange-400">W±</span><div class="text-xs text-slate-500 mt-1">${t('boson-w-direction')}</div></div>`;
        ui.gluonColorSelector.classList.add('hidden');
    } else if (toolType === 'gluon') {
        container.innerHTML = `<span class="text-slate-500 text-sm font-mono">${t('tool-gluon')} (g)</span>`;
        ui.gluonColorSelector.classList.remove('hidden');
        ui.selGluonColor.innerHTML = GLUON_COLORS.map((gc, idx) => `<option value="${idx}" ${idx===(props.gluonColor||0)?'selected':''}>${gc.display}</option>`).join('');
        if (props.gluonColor === null || props.gluonColor === undefined) { currentProps.gluonColor = 0; }
    } else {
        container.innerHTML = `<span class="text-slate-500 text-sm font-mono">${getParticleSymbol(props)}</span>`;
        ui.gluonColorSelector.classList.add('hidden');
    }
}

function syncChanges() {
    if (selectedShapeIds.size > 0) {
        shapes.forEach(s => {
            if (selectedShapeIds.has(s.id)) {
                if (s.type === 'fermion') {
                    s.props.group = currentProps.group;
                    s.props.particleId = currentProps.particleId;
                    s.props.color = currentProps.color;
                }
                if (s.type === 'gluon') {
                    s.props.gluonColor = currentProps.gluonColor;
                }
            }
        });
        draw();
        if (selectedShapeIds.size === 1) { updateUI(); }
    } else {
        updatePhysicsPreview(currentProps);
    }
}

function updatePhysicsPreview(props) {
    const p = getParticleData(props);
    if (!p) {
        ['Q', 'L', 'B', 'Spin', 'Parity'].forEach(id => ui[`preview${id}`].textContent = '-');
        return;
    }
    
    const isAnti = props ? props.isAnti || false : false;
    let q = p.charge, l = p.lepton, b = p.baryon, spin = p.spin || 0, parity = p.parity || 1;

    if (props && props.category === 'fermion' && isAnti) {
        q = -q; l = -l; b = -b; parity = -parity;
    }
    
    ui.previewQ.textContent = q.toFixed(2);
    ui.previewL.textContent = l.toFixed(2);
    ui.previewB.textContent = b.toFixed(2);
    ui.previewSpin.textContent = spin;
    ui.previewParity.textContent = parity > 0 ? '+1' : '-1';
}

document.getElementById('btn-clear').onclick = () => {
    if (selectedShapeIds.size > 0) {
        shapes = shapes.filter(s => !selectedShapeIds.has(s.id));
        selectedShapeIds.clear();
    } else {
        if (confirm(t('confirm-clear'))) {
            shapes = [];
            document.getElementById('error-layer').innerHTML = '';
        }
    }
    draw();
    updateUI();
};

let toastTimer;
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    
    toastMessage.textContent = message;
    
    toast.className = 'fixed top-16 md:top-20 left-1/2 -translate-x-1/2 text-white px-4 py-2 rounded-lg text-sm shadow-xl z-50 transition-all duration-300 opacity-0 -translate-y-10 pointer-events-none max-w-md';
    if (type === 'success') toast.classList.add('bg-green-600', 'border-green-400');
    else if (type === 'error') toast.classList.add('bg-red-600', 'border-red-400');
    else if (type === 'warning') toast.classList.add('bg-yellow-600', 'border-yellow-400');
    else toast.classList.add('bg-slate-800', 'border-slate-600');

    toast.classList.remove('opacity-0', '-translate-y-10');
    
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.add('opacity-0', '-translate-y-10');
    }, 3000);
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' && selectedShapeIds.size > 0) {
        shapes = shapes.filter(s => !selectedShapeIds.has(s.id));
        selectedShapeIds.clear();
        draw();
        updateUI();
    }
    if (e.key === 'Escape') {
        selectedShapeIds.clear();
        isDrawing = false;
        draftLine = null;
        draw();
        updateUI();
        document.getElementById('help-panel').classList.add('hidden');
    }
    if (e.ctrlKey && e.key === 'a') { e.preventDefault(); shapes.forEach(s => selectedShapeIds.add(s.id)); draw(); updateUI(); }
    if (e.ctrlKey && e.key === 'z' && !e.shiftKey) { e.preventDefault(); if (shapes.length > 0) { shapes.pop(); draw(); } }
    if (e.key === 'h' || e.key === 'H' || e.key === 'F1') { e.preventDefault(); document.getElementById('help-panel').classList.toggle('hidden'); }
});

document.getElementById('btn-help').onclick = () => document.getElementById('help-panel').classList.toggle('hidden');
document.getElementById('btn-close-help').onclick = () => document.getElementById('help-panel').classList.add('hidden');

// --- 新增的事件监听 ---
document.getElementById('btn-lang').onclick = () => {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    localStorage.setItem('language', currentLang);
    updateLanguage();
};

document.getElementById('btn-back-to-start').onclick = () => {
    window.location.href = 'start.html';
};

// ===== AI 自动生成功能 =====

// 打开 AI 面板
document.getElementById('btn-ai-generate')?.addEventListener('click', () => {
    const aiPanel = document.getElementById('ai-panel');
    aiPanel.classList.remove('hidden');
    
    // 从 localStorage 加载已保存的 API Key
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) {
        document.getElementById('input-api-key').value = savedKey;
        window.FeynmanDiagramGenerator.setAPIKey(savedKey);
    }
});

// 关闭 AI 面板
document.getElementById('btn-close-ai')?.addEventListener('click', () => {
    document.getElementById('ai-panel').classList.add('hidden');
});

// 保存 API Key
document.getElementById('btn-save-key')?.addEventListener('click', () => {
    const apiKey = document.getElementById('input-api-key').value.trim();
    if (!apiKey) {
        showToast('请输入 API Key', 'error');
        return;
    }
    
    localStorage.setItem('gemini_api_key', apiKey);
    window.FeynmanDiagramGenerator.setAPIKey(apiKey);
    showToast('✅ API Key 已保存', 'success');
});

// 生成费曼图
document.getElementById('btn-generate-diagram')?.addEventListener('click', async () => {
    const reactionInput = document.getElementById('input-reaction').value.trim();
    if (!reactionInput) {
        showToast('请输入反应式', 'error');
        return;
    }
    
    const apiKey = localStorage.getItem('gemini_api_key');
    if (!apiKey) {
        showToast('请先保存 API Key', 'error');
        return;
    }
    
    // 检查是否生成所有图表
    const generateAll = document.getElementById('checkbox-generate-all')?.checked || false;
    
    // 显示加载状态
    document.getElementById('ai-loading').classList.remove('hidden');
    document.getElementById('ai-result').classList.add('hidden');
    
    try {
        // 调用 AI 生成函数
        const canvasWidth = canvas.width / dpr;
        const canvasHeight = canvas.height / dpr;
        
        const result = await window.FeynmanDiagramGenerator.generateFeynmanDiagram(
            reactionInput,
            canvasWidth,
            canvasHeight,
            { generateAll }
        );
        
        // 如果生成了多个图表，为每个图表创建一个新画布
        if (Array.isArray(result)) {
            console.log(`📊 生成了 ${result.length} 个图表，为每个创建新画布...`);
            
            const canvasManager = window.getCanvasManager();
            if (!canvasManager) {
                throw new Error('画布管理器未初始化');
            }
            
            // 🔧 暂时禁用自动保存，避免在批量创建时冲突
            window._disableAutoSave = true;
            
            try {
                // 为每个图表创建画布并加载
                for (let i = 0; i < result.length; i++) {
                    const diagramResult = result[i];
                    const canvasName = `${reactionInput.replace(/\$/g, '')} - ${diagramResult.diagramName}`;
                    
                    if (i === 0) {
                        // 第一个图表：更新当前画布名称和内容
                        canvasManager.renameCanvas(canvasManager.currentCanvasIndex, canvasName);
                        canvasManager.updateCurrentCanvas(diagramResult.shapes);
                    } else {
                        // 其他图表：创建新画布（shapes 已经在参数中传入）
                        canvasManager.createNewCanvas(canvasName, diagramResult.shapes);
                    }
                }
                
                // 加载第一个图表到当前画布并绘制
                shapes = [...result[0].shapes]; // 深拷贝
                selectedShapeIds.clear();
                draw();
                
            } finally {
                // 🔧 恢复自动保存
                window._disableAutoSave = false;
            }
            
            // 触发 shapes 更新事件
            window.dispatchEvent(new CustomEvent('shapesUpdated', { detail: { shapes } }));
            
            showToast(`✅ 成功生成 ${result.length} 个费曼图，已分别保存到不同画布`, 'success');
            
            // 显示结果
            document.getElementById('ai-loading').classList.add('hidden');
            document.getElementById('ai-result').classList.remove('hidden');
            document.getElementById('ai-explanation').innerHTML = `
                <div class="mb-2">
                    <span class="font-bold text-cyan-400">相互作用类型：</span>
                    <span class="text-white">${result[0].interactionType}</span>
                </div>
                <div class="mb-2">
                    <span class="font-bold text-green-400">生成图表数：</span>
                    <span class="text-white">${result.length} 个</span>
                    <span class="text-slate-400 text-xs ml-2">(${result.map(r => r.diagramName).join(', ')})</span>
                </div>
                <div>
                    <span class="font-bold text-cyan-400">物理解释：</span>
                    <div class="text-slate-300 mt-1">${result[0].explanation}</div>
                </div>
            `;
            
        } else {
            // 单个图表
            shapes = [];
            selectedShapeIds.clear();
            shapes.push(...result.shapes);
            draw();
            
            // 触发 shapes 更新事件
            window.dispatchEvent(new CustomEvent('shapesUpdated', { detail: { shapes } }));
            
            showToast('✅ AI 生成成功！已自动绘制到画布', 'success');
            
            // 显示结果
            document.getElementById('ai-loading').classList.add('hidden');
            document.getElementById('ai-result').classList.remove('hidden');
            document.getElementById('ai-explanation').innerHTML = `
                <div class="mb-2">
                    <span class="font-bold text-cyan-400">相互作用类型：</span>
                    <span class="text-white">${result.interactionType}</span>
                </div>
                <div>
                    <span class="font-bold text-cyan-400">物理解释：</span>
                    <div class="text-slate-300 mt-1">${result.explanation}</div>
                </div>
            `;
        }
        
        // 3秒后关闭面板
        setTimeout(() => {
            document.getElementById('ai-panel').classList.add('hidden');
        }, 3000);
        
    } catch (error) {
        console.error('AI 生成失败:', error);
        document.getElementById('ai-loading').classList.add('hidden');
        showToast('❌ 生成失败: ' + error.message, 'error');
    }
});

// ESC 键关闭 AI 面板
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.getElementById('ai-panel')?.classList.add('hidden');
    }
});

// ==================== 画布管理辅助函数 ====================

/**
 * 加载 shapes 到画布
 */
window.loadShapes = function(newShapes) {
    shapes = JSON.parse(JSON.stringify(newShapes)); // 深拷贝
    selectedShapeIds.clear();
    draw();
    console.log(`✅ 加载了 ${shapes.length} 个形状到画布`);
};

/**
 * 清空所有形状
 */
window.clearAllShapes = function() {
    shapes = [];
    selectedShapeIds.clear();
    draw();
    
    // 触发 shapes 更新事件
    window.dispatchEvent(new CustomEvent('shapesUpdated', { detail: { shapes } }));
};

/**
 * 获取当前 shapes（供 canvas-manager 保存）
 */
window.getCurrentShapes = function() {
    return shapes;
};
