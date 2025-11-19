// ===== Gemini 2.5 Pro Integration for Automatic Feynman Diagram Generation =====

/**
 * 这个模块负责：
 * 1. 与 Gemini 2.5 Pro API 通信
 * 2. 将用户输入的反应式转换为费曼图数据
 * 3. 自动绘制最低阶（lowest degree）费曼图
 */

// Gemini API 配置
const GEMINI_CONFIG = {
    apiKey: '', // 用户需要填入自己的 API Key
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent',
    model: 'gemini-2.5-pro'
};

/**
 * 物理反应数据库 - 用于参考
 */
const REACTION_DATABASE = {
    // QED 反应
    'electron_positron_annihilation': {
        initial: ['e-', 'e+'],
        final: ['photon', 'photon'],
        process: 'e⁻ + e⁺ → γ + γ',
        interaction: 'electromagnetic',
        diagram_count: 2 // t-channel and u-channel
    },
    'muon_pair_production': {
        initial: ['e-', 'e+'],
        final: ['mu-', 'mu+'],
        process: 'e⁻ + e⁺ → μ⁻ + μ⁺',
        interaction: 'electromagnetic',
        diagram_count: 1 // s-channel
    },
    'bhabha_scattering': {
        initial: ['e-', 'e+'],
        final: ['e-', 'e+'],
        process: 'e⁻ + e⁺ → e⁻ + e⁺',
        interaction: 'electromagnetic',
        diagram_count: 2 // s-channel and t-channel
    },
    'compton_scattering': {
        initial: ['photon', 'e-'],
        final: ['photon', 'e-'],
        process: 'γ + e⁻ → γ + e⁻',
        interaction: 'electromagnetic',
        diagram_count: 2
    },
    // 弱相互作用
    'beta_decay': {
        initial: ['n'],
        final: ['p', 'e-', 'anti-nu_e'],
        process: 'n → p + e⁻ + ν̄ₑ',
        interaction: 'weak',
        diagram_count: 1
    },
    'muon_decay': {
        initial: ['mu-'],
        final: ['e-', 'anti-nu_e', 'nu_mu'],
        process: 'μ⁻ → e⁻ + ν̄ₑ + νμ',
        interaction: 'weak',
        diagram_count: 1
    },
    // 强相互作用
    'quark_gluon_vertex': {
        initial: ['u', 'gluon'],
        final: ['u'],
        process: 'u + g → u',
        interaction: 'strong',
        diagram_count: 1
    }
};

/**
 * 粒子符号映射表 (用于解析用户输入)
 * 使用 LaTeX 格式：$e-$, $e+$, $mu-$, $mu+$ 等
 */
const PARTICLE_SYMBOLS = {
    // 轻子 - LaTeX 格式
    '$e-$': { id: 'e', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'e⁻', latex: 'e^-' },
    '$e+$': { id: 'e', category: 'fermion', group: 'lepton', isAnti: true, symbol: 'e⁺', latex: 'e^+' },
    '$mu-$': { id: 'mu', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'μ⁻', latex: '\\mu^-' },
    '$mu+$': { id: 'mu', category: 'fermion', group: 'lepton', isAnti: true, symbol: 'μ⁺', latex: '\\mu^+' },
    '$tau-$': { id: 'tau', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'τ⁻', latex: '\\tau^-' },
    '$tau+$': { id: 'tau', category: 'fermion', group: 'lepton', isAnti: true, symbol: 'τ⁺', latex: '\\tau^+' },
    '$nu_e$': { id: 'nu_e', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'νₑ', latex: '\\nu_e' },
    '$anti-nu_e$': { id: 'nu_e', category: 'fermion', group: 'lepton', isAnti: true, symbol: 'ν̄ₑ', latex: '\\bar{\\nu}_e' },
    '$nu_mu$': { id: 'nu_mu', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'νμ', latex: '\\nu_\\mu' },
    '$nu_tau$': { id: 'nu_tau', category: 'fermion', group: 'lepton', isAnti: false, symbol: 'ντ', latex: '\\nu_\\tau' },
    
    // 夸克 - LaTeX 格式（默认红色）
    '$u$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'u', color: 'red', latex: 'u' },
    '$ubar$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'ū', color: 'anti-red', latex: '\\bar{u}' },
    '$d$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'd', color: 'red', latex: 'd' },
    '$dbar$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'd̄', color: 'anti-red', latex: '\\bar{d}' },
    '$c$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'c', color: 'red', latex: 'c' },
    '$cbar$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'c̄', color: 'anti-red', latex: '\\bar{c}' },
    '$s$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 's', color: 'red', latex: 's' },
    '$sbar$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 's̄', color: 'anti-red', latex: '\\bar{s}' },
    '$t$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 't', color: 'red', latex: 't' },
    '$tbar$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 't̄', color: 'anti-red', latex: '\\bar{t}' },
    '$b$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'b', color: 'red', latex: 'b' },
    '$bbar$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'b̄', color: 'anti-red', latex: '\\bar{b}' },
    
    // 夸克 - 带颜色下标 (red/green/blue)
    '$u_red$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'u', color: 'red', latex: 'u^r' },
    '$u_green$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'u', color: 'green', latex: 'u^g' },
    '$u_blue$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'u', color: 'blue', latex: 'u^b' },
    '$d_red$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'd', color: 'red', latex: 'd^r' },
    '$d_green$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'd', color: 'green', latex: 'd^g' },
    '$d_blue$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'd', color: 'blue', latex: 'd^b' },
    '$c_red$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'c', color: 'red', latex: 'c^r' },
    '$c_green$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'c', color: 'green', latex: 'c^g' },
    '$c_blue$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 'c', color: 'blue', latex: 'c^b' },
    '$s_red$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 's', color: 'red', latex: 's^r' },
    '$s_green$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 's', color: 'green', latex: 's^g' },
    '$s_blue$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 's', color: 'blue', latex: 's^b' },
    '$t_red$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 't', color: 'red', latex: 't^r' },
    '$t_green$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 't', color: 'green', latex: 't^g' },
    '$t_blue$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: false, symbol: 't', color: 'blue', latex: 't^b' },
    '$b_red$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'b', color: 'red', latex: 'b^r' },
    '$b_green$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'b', color: 'green', latex: 'b^g' },
    '$b_blue$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: false, symbol: 'b', color: 'blue', latex: 'b^b' },
    
    // 反夸克 - 带反色荷
    '$ubar_anti-red$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'ū', color: 'anti-red', latex: '\\bar{u}^{\\bar{r}}' },
    '$ubar_anti-green$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'ū', color: 'anti-green', latex: '\\bar{u}^{\\bar{g}}' },
    '$ubar_anti-blue$': { id: 'u', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'ū', color: 'anti-blue', latex: '\\bar{u}^{\\bar{b}}' },
    '$dbar_anti-red$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'd̄', color: 'anti-red', latex: '\\bar{d}^{\\bar{r}}' },
    '$dbar_anti-green$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'd̄', color: 'anti-green', latex: '\\bar{d}^{\\bar{g}}' },
    '$dbar_anti-blue$': { id: 'd', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'd̄', color: 'anti-blue', latex: '\\bar{d}^{\\bar{b}}' },
    '$cbar_anti-red$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'c̄', color: 'anti-red', latex: '\\bar{c}^{\\bar{r}}' },
    '$cbar_anti-green$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'c̄', color: 'anti-green', latex: '\\bar{c}^{\\bar{g}}' },
    '$cbar_anti-blue$': { id: 'c', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 'c̄', color: 'anti-blue', latex: '\\bar{c}^{\\bar{b}}' },
    '$sbar_anti-red$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 's̄', color: 'anti-red', latex: '\\bar{s}^{\\bar{r}}' },
    '$sbar_anti-green$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 's̄', color: 'anti-green', latex: '\\bar{s}^{\\bar{g}}' },
    '$sbar_anti-blue$': { id: 's', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 's̄', color: 'anti-blue', latex: '\\bar{s}^{\\bar{b}}' },
    '$tbar_anti-red$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 't̄', color: 'anti-red', latex: '\\bar{t}^{\\bar{r}}' },
    '$tbar_anti-green$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 't̄', color: 'anti-green', latex: '\\bar{t}^{\\bar{g}}' },
    '$tbar_anti-blue$': { id: 't', category: 'fermion', group: 'quark_u', isAnti: true, symbol: 't̄', color: 'anti-blue', latex: '\\bar{t}^{\\bar{b}}' },
    '$bbar_anti-red$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'b̄', color: 'anti-red', latex: '\\bar{b}^{\\bar{r}}' },
    '$bbar_anti-green$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'b̄', color: 'anti-green', latex: '\\bar{b}^{\\bar{g}}' },
    '$bbar_anti-blue$': { id: 'b', category: 'fermion', group: 'quark_d', isAnti: true, symbol: 'b̄', color: 'anti-blue', latex: '\\bar{b}^{\\bar{b}}' },
    
    // 胶子 - 带色荷组合
    '$g_rg$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 0, latex: 'g_{r\\bar{g}}' },  // red-antigreen
    '$g_rb$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 1, latex: 'g_{r\\bar{b}}' },  // red-antiblue
    '$g_gr$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 2, latex: 'g_{g\\bar{r}}' },  // green-antired
    '$g_gb$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 3, latex: 'g_{g\\bar{b}}' },  // green-antiblue
    '$g_br$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 4, latex: 'g_{b\\bar{r}}' },  // blue-antired
    '$g_bg$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 5, latex: 'g_{b\\bar{g}}' },  // blue-antigreen
    
    // 玻色子 - LaTeX 格式
    '$photon$': { id: 'photon', category: 'boson', symbol: 'γ', latex: '\\gamma' },
    '$gamma$': { id: 'photon', category: 'boson', symbol: 'γ', latex: '\\gamma' },
    '$g$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 0, latex: 'g' },
    '$gluon$': { id: 'gluon', category: 'boson', symbol: 'g', gluonColor: 0, latex: 'g' },
    '$W+$': { id: 'w_plus', category: 'boson', symbol: 'W⁺', latex: 'W^+' },
    '$W-$': { id: 'w_minus', category: 'boson', symbol: 'W⁻', latex: 'W^-' },
    '$Z$': { id: 'z', category: 'boson', symbol: 'Z⁰', latex: 'Z^0' },
    '$Z0$': { id: 'z', category: 'boson', symbol: 'Z⁰', latex: 'Z^0' },
    '$H$': { id: 'higgs', category: 'boson', symbol: 'H', latex: 'H' },
    '$higgs$': { id: 'higgs', category: 'boson', symbol: 'H', latex: 'H' }
};

/**
 * 解析用户输入的反应式
 * 支持格式（LaTeX）：
 * - "$e-$ + $e+$ $->$ $mu-$ + $mu+$"
 * - "$gamma$ + $e-$ $->$ $gamma$ + $e-$"
 */
function parseReaction(input) {
    // 标准化输入 - 称除多余空格
    input = input.trim();
    
    // 分割初态和末态 - 使用 $->$ 作为分隔符
    const parts = input.split('$->$');
    if (parts.length !== 2) {
        throw new Error('反应式格式错误。请使用 "$e-$ + $e+$ $->$ $mu-$ + $mu+$" 格式（LaTeX）');
    }
    
    // 提取粒子符号的函数 - 匹配所有 $...$ 模式
    function extractParticles(str) {
        const particles = [];
        const regex = /\$([^$]+)\$/g;
        let match;
        while ((match = regex.exec(str)) !== null) {
            const particle = '$' + match[1] + '$';
            particles.push(particle);
        }
        return particles;
    }
    
    const initialState = extractParticles(parts[0]);
    const finalState = extractParticles(parts[1]);
    
    // 解析粒子
    const initial = initialState.map(parseParticle);
    const final = finalState.map(parseParticle);
    
    return {
        initial,
        final,
        originalInput: input
    };
}

/**
 * 解析单个粒子符号（LaTeX 格式）
 */
function parseParticle(symbol) {
    const cleanSymbol = symbol.trim();
    
    if (!PARTICLE_SYMBOLS[cleanSymbol]) {
        // 提供更友好的错误信息
        const supportedParticles = Object.keys(PARTICLE_SYMBOLS).slice(0, 10).join(', ');
        throw new Error(`未知粒子符号: "${symbol}". 支持的粒子: ${supportedParticles}...`);
    }
    
    return {
        ...PARTICLE_SYMBOLS[cleanSymbol],
        originalSymbol: symbol
    };
}

/**
 * 构建 Gemini API prompt
 */
function buildGeminiPrompt(reaction, canvasWidth, canvasHeight) {
    // 构建粒子描述，包含颜色信息
    const initialParticles = reaction.initial.map(p => {
        if (p.color) {
            return `${p.symbol}(${p.color})`;
        }
        return p.symbol;
    }).join(' + ');
    
    const finalParticles = reaction.final.map(p => {
        if (p.color) {
            return `${p.symbol}(${p.color})`;
        }
        return p.symbol;
    }).join(' + ');
    
    const prompt = `You are an expert in particle physics and Feynman diagrams. Generate ALL LOWEST-ORDER (tree-level) Feynman diagrams for the given reaction.

**Reaction:** ${initialParticles} → ${finalParticles}

**DIMENSION 1: CHAIN OF THOUGHT (PHYSICS ANALYSIS)**
Before generating the JSON, you MUST analyze the reaction step-by-step in a "physics_analysis" field:
1.  Identify the interaction type (EM, Weak, Strong).
2.  Check conservation laws (Charge, Lepton number, Baryon number, Color charge).
3.  Determine the topology (s-channel, t-channel, u-channel, or decay).
4.  List the vertices and the particles entering/leaving each vertex.
5.  **For QCD:** Explicitly trace the color flow (e.g., "Red quark emits Red-AntiBlue gluon, becoming Blue quark").

**DIMENSION 2: TOPOLOGICAL GRAPH VECTOR (NO COORDINATES)**
-   **Do NOT output coordinates (x, y).**
-   Only output the connection logic (nodes and edges).
-   **Nodes:** Define vertices with IDs and types ("initial", "final", "interaction").
-   **Edges:** Define connections between nodes with particle properties.

**DIMENSION 3: QCD COLOR RULES**
-   **Color Conservation:** The net color entering a vertex MUST equal the net color leaving it.
-   **Gluons:** Must carry a color and an anti-color (e.g., "red-antiblue").
-   **Quarks:** Must have a single color (red, green, blue).
-   **Anti-Quarks:** Must have a single anti-color (anti-red, anti-green, anti-blue).

**CRITICAL RULES:**
1.  **ONLY generate LOWEST-ORDER (tree-level) diagrams** - NO loop diagrams, NO radiative corrections
2.  **For flow diagrams: Minimize additional gluon exchanges** - Keep diagrams simple.
3.  **Generate ALL possible tree-level channels.**
4.  **Put each diagram in the "diagrams" array.**

**STRICT JSON SCHEMA:**
You MUST follow this exact JSON structure. Do not deviate.
\\\`\\\`\\\`json
{
  "physics_analysis": "...",
  "valid": true,
  "interaction_type": "electromagnetic" | "weak" | "strong",
  "diagrams": [
    {
      "name": "diagram-name",
      "nodes": [
        { "id": "n1", "type": "initial" | "final" | "interaction" }
      ],
      "edges": [
        {
          "source": "n1",
          "target": "n2",
          "particle": {
            "id": "particle_id", // MUST be one of the allowed IDs below
            "category": "fermion" | "boson",
            "group": "lepton" | "quark_u" | "quark_d" | "boson", // REQUIRED. See Reference Table below.
            "isAnti": boolean,
            "isVirtual": boolean,
            "color": "red" | "green" | "blue" | "anti-red" | "anti-green" | "anti-blue" | "red-antiblue" ... // Optional
          }
        }
      ],
      "loops": [ // Optional, only for penguin diagrams
        {
          "vertices": ["v1", "v2", "v3"], // MUST use "vertices" key, NOT "arc"
          "type": "penguin_loop"
        }
      ]
    }
  ]
}
\\\`\\\`\\\`

**PARTICLE REFERENCE TABLE (STRICTLY FOLLOW THIS):**
You MUST use the exact "id", "category", and "group" from this table.

| Particle Name | ID | Category | Group |
| :--- | :--- | :--- | :--- |
| Electron, Muon, Tau | e, mu, tau | fermion | lepton |
| Neutrinos | nu_e, nu_mu, nu_tau | fermion | lepton |
| Up, Charm, Top Quarks | u, c, t | fermion | quark_u |
| Down, Strange, Bottom Quarks | d, s, b | fermion | quark_d |
| Photon, Gluon, Z, Higgs | photon, gluon, z, higgs | boson | boson |
| W Bosons | w_plus, w_minus | boson | boson |

**PHYSICS CORRECTION RULES:**
-   **Anti-Quark Weak Decay:** Ensure W boson charge is correct.
    -   Example: b_bar (+1/3) -> c_bar (-2/3) + W_plus (+1). (Correct)
    -   Example: b_bar (+1/3) -> c_bar (-2/3) + W_minus (-1). (INCORRECT - Charge violation)
-   **Penguin Loops:** If the process is b -> s (FCNC), use a penguin loop. The loop usually involves a virtual top quark (t) and a W boson.

**DIAGRAM TYPE SELECTION:**
-   **DEFAULT: Tree-level diagrams ONLY**
-   **EXCEPTION: For FCNC processes ONLY** (b→s transitions with γ/Z): Use penguin loop diagram.

**Generate the complete JSON now for:** ${initialParticles} → ${finalParticles}`;

    return prompt;
}

/**
 * 修复 Gemini 返回的不完整 JSON
 */
function fixIncompleteJSON(jsonString) {
    let fixed = jsonString.trim();
    
    // 1. 移除多余的逗号
    fixed = fixed.replace(/,(\s*[}\]])/g, '$1');
    
    // 2. 检查并补全缺失的右括号
    const openBraces = (fixed.match(/\{/g) || []).length;
    const closeBraces = (fixed.match(/\}/g) || []).length;
    const openBrackets = (fixed.match(/\[/g) || []).length;
    const closeBrackets = (fixed.match(/\]/g) || []).length;
    
    // 补全缺失的 ]
    if (openBrackets > closeBrackets) {
        fixed += ']'.repeat(openBrackets - closeBrackets);
    }
    
    // 补全缺失的 }
    if (openBraces > closeBraces) {
        fixed += '}'.repeat(openBraces - closeBraces);
    }
    
    // 3. 处理未闭合的字符串（最常见的问题）
    // 如果在某个位置出现解析错误，尝试在该位置添加引号
    
    return fixed;
}

/**
 * 调用 Gemini API
 */
async function callGeminiAPI(prompt) {
    if (!GEMINI_CONFIG.apiKey) {
        throw new Error('请先设置 Gemini API Key！在 gemini-integration.js 中填入 GEMINI_CONFIG.apiKey');
    }
    
    try {
        const response = await fetch(`${GEMINI_CONFIG.endpoint}?key=${GEMINI_CONFIG.apiKey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                contents: [{
                    parts: [{
                        text: prompt
                    }]
                }],
                generationConfig: {
                    temperature: 0.2, // 低温度保证物理准确性
                    topK: 40,
                    topP: 0.95,
                    maxOutputTokens: 99999, // 减少 token 数避免 JSON 过长
                    responseMimeType: "application/json", // 要求返回纯 JSON
                }
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Gemini API 错误: ${error.error?.message || response.statusText}`);
        }
        
        const data = await response.json();
        
        // 检查响应结构
        if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
            console.error('Gemini 响应结构异常:', JSON.stringify(data, null, 2));
            throw new Error('Gemini 返回的数据结构不完整');
        }
        
        const text = data.candidates[0].content.parts[0].text;
        console.log('📥 Gemini 原始返回:', text);
        
        // 提取 JSON (Gemini 可能用 ```json ``` 包裹)
        const jsonMatch = text.match(/```json\\s*([\\s\\S]*?)\\s*```/);
        let jsonString;

        if (jsonMatch && jsonMatch[1]) {
            jsonString = jsonMatch[1].trim();
        } else {
            // 如果没有找到 ```json ```, 假定整个返回就是 JSON
            jsonString = text.trim();
        }
        
        if (!jsonString) {
            console.error('❌ Gemini 返回的文本不包含 JSON:', text);
            throw new Error('Gemini 返回的格式不正确，未找到 JSON 数据');
        }
        
        // 🔧 尝试修复常见的 JSON 错误
        try {
            const parsedData = JSON.parse(jsonString);
            console.log('✅ Gemini 返回的数据:', parsedData);
            return parsedData;
        } catch (parseError) {
            console.warn('⚠️ JSON 解析失败，尝试修复...', parseError.message);
            console.log('原始 JSON 前 1000 字符:', jsonString.substring(0, 1000));
            console.log('原始 JSON 后 500 字符:', jsonString.substring(jsonString.length - 500));
            
            // 使用修复函数
            const fixedJSON = fixIncompleteJSON(jsonString);
            
            try {
                const parsedData = JSON.parse(fixedJSON);
                console.log('✅ 修复后的数据:', parsedData);
                return parsedData;
            } catch (retryError) {
                console.error('❌ 修复失败');
                console.error('修复后的 JSON 前 1000 字符:', fixedJSON.substring(0, 1000));
                console.error('修复后的 JSON 后 500 字符:', fixedJSON.substring(fixedJSON.length - 500));
                throw new Error(`JSON 解析失败: ${parseError.message}\n\n可能原因：\n1. Gemini 返回的数据过长被截断\n2. JSON 格式不正确\n\n请尝试：\n- 简化反应式（减少粒子数量）\n- 重新生成\n- 检查浏览器控制台查看详细日志`);
            }
        }
    } catch (error) {
        console.error('Gemini API 调用失败:', error);
        throw error;
    }
}

/**
 * 将 Gemini 返回的 JSON 转换为 FeymanForge 的 shapes 格式
 */
function convertToShapes(geminiResponse, canvasWidth, canvasHeight) {
    const shapes = [];
    
    if (!geminiResponse.valid) {
        throw new Error('Gemini 判断该反应物理上不合法：' + geminiResponse.explanation);
    }
    
    // 检查是否有图表数据
    if (!geminiResponse.diagrams || geminiResponse.diagrams.length === 0) {
        throw new Error('Gemini 未返回有效的费曼图数据');
    }
    
    console.log(`📊 Gemini 返回了 ${geminiResponse.diagrams.length} 个费曼图`);
    
    // 🎨 如果有多个图表,选择第一个(用户可以通过 canvas 切换查看其他图)
    const diagram = geminiResponse.diagrams[0];
    
    if (geminiResponse.diagrams.length > 1) {
        console.log(`✨ 可选图表: ${geminiResponse.diagrams.map(d => d.name).join(', ')}`);
        console.log(`📌 当前显示: ${diagram.name}`);
    }
    
    // 检查图表是否有必要的数据
    // 新格式：nodes 和 edges
    if (!diagram.nodes || !diagram.edges) {
        // 兼容旧格式（如果 Gemini 偶尔返回旧格式）
        if (diagram.vertices && diagram.lines) {
            console.warn('⚠️ Gemini 返回了旧格式数据 (vertices/lines)，尝试兼容...');
            // 将旧格式转换为新格式
            diagram.nodes = diagram.vertices.map(v => ({ id: v.id, type: v.type || 'interaction' }));
            diagram.edges = diagram.lines.map(l => ({ source: l.from, target: l.to, particle: l.particle }));
        } else {
            throw new Error('费曼图缺少节点或边数据 (nodes/edges)');
        }
    }

    // 🔧 FORCE PENGUIN TOPOLOGY (User Request: W-W-Quark triangle with Z emitted from W-W vertex)
    // This overrides Gemini's physics choice to ensure the visual style matches the user's expectation
    if (diagram.loops && diagram.loops.length > 0) {
        diagram.loops.forEach(loop => {
            if (loop.type === 'penguin_loop' && loop.vertices && loop.vertices.length === 3) {
                const vIds = loop.vertices;
                
                // Find emission vertex (connected to Z/Photon/Gluon)
                let emissionVertexId = null;
                diagram.edges.forEach(e => {
                    if (vIds.includes(e.source) && !vIds.includes(e.target)) {
                        if (['z', 'photon', 'gluon', 'higgs'].includes(e.particle.id)) {
                            emissionVertexId = e.source;
                        }
                    }
                });

                if (emissionVertexId) {
                    console.log(`🔧 Enforcing Penguin Topology: Z/Photon emission at ${emissionVertexId}`);
                    const baseVertices = vIds.filter(id => id !== emissionVertexId);
                    
                    if (baseVertices.length === 2) {
                        // Determine if we are dealing with particles or anti-particles based on input
                        let isAntiContext = false;
                        diagram.edges.forEach(e => {
                            if (baseVertices.includes(e.target) && e.particle.category === 'fermion') {
                                if (e.particle.isAnti) isAntiContext = true;
                            }
                        });

                        diagram.edges.forEach(e => {
                            const isLoopEdge = vIds.includes(e.source) && vIds.includes(e.target);
                            if (!isLoopEdge) return;

                            const connectedToEmission = e.source === emissionVertexId || e.target === emissionVertexId;
                            
                            if (connectedToEmission) {
                                // Sides of the triangle -> W Boson
                                e.particle.id = isAntiContext ? 'w_plus' : 'w_minus';
                                e.particle.category = 'boson';
                                e.particle.group = 'boson';
                                delete e.particle.color;
                                delete e.particle.isAnti;
                            } else {
                                // Base of the triangle -> Heavy Quark (Top)
                                e.particle.id = 't';
                                e.particle.category = 'fermion';
                                e.particle.group = 'quark_u';
                                e.particle.isAnti = isAntiContext;
                                e.particle.color = isAntiContext ? 'anti-red' : 'red';
                            }
                        });
                    }
                }
            }
        });
    }

    // === 使用 FeynmanLayoutEngine 计算坐标 ===
    console.log('📐 正在使用 FeynmanLayoutEngine 计算布局...');
    if (!window.FeynmanLayoutEngine) {
        throw new Error('FeynmanLayoutEngine 未加载');
    }
    
    const layoutEngine = new window.FeynmanLayoutEngine(canvasWidth, canvasHeight);
    const positions = layoutEngine.calculateLayout({ nodes: diagram.nodes, edges: diagram.edges });

    // 🔧 Spectator Line Handling: Move spectators to edges to avoid crossing the diagram
    // Spectators are direct connections between initial and final states
    const spectatorEdges = diagram.edges.filter(e => {
        const sourceNode = diagram.nodes.find(n => n.id === e.source);
        const targetNode = diagram.nodes.find(n => n.id === e.target);
        return sourceNode && targetNode && 
               sourceNode.type === 'initial' && targetNode.type === 'final';
    });

    if (spectatorEdges.length > 0) {
        console.log(`🔧 Found ${spectatorEdges.length} spectator edges, moving them to boundaries...`);
        const topY = 50;
        const bottomY = canvasHeight - 50;
        
        spectatorEdges.forEach((edge, idx) => {
            // Alternate between top and bottom to avoid overlapping multiple spectators
            const isTop = idx % 2 === 0;
            const y = isTop ? (topY + idx * 30) : (bottomY - (idx * 30));
            
            if (positions[edge.source]) positions[edge.source].y = y;
            if (positions[edge.target]) positions[edge.target].y = y;
        });
    }
    
    // 🔧 Post-processing: Ensure nodes are not too close (Collision Resolution)
    // This prevents "monster vertices" where multiple nodes collapse into one
    const MIN_DISTANCE = 40; // Minimum distance between nodes
    const nodeIds = Object.keys(positions);
    
    for (let i = 0; i < nodeIds.length; i++) {
        for (let j = i + 1; j < nodeIds.length; j++) {
            const id1 = nodeIds[i];
            const id2 = nodeIds[j];
            const p1 = positions[id1];
            const p2 = positions[id2];
            
            if (!p1 || !p2) continue;
            
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            const dist = Math.hypot(dx, dy);
            
            if (dist < MIN_DISTANCE) {
                console.warn(`⚠️ Nodes ${id1} and ${id2} are too close (${dist.toFixed(1)}px), separating...`);
                // Simple separation: move p2 away from p1
                // If they are at the exact same position, move p2 randomly
                let moveX = dx === 0 ? (Math.random() - 0.5) : dx;
                let moveY = dy === 0 ? (Math.random() - 0.5) : dy;
                
                // Normalize and scale
                const len = Math.hypot(moveX, moveY) || 1;
                moveX = (moveX / len) * (MIN_DISTANCE - dist + 5);
                moveY = (moveY / len) * (MIN_DISTANCE - dist + 5);
                
                // Apply to p2 (and maybe p1 in opposite direction)
                // Only move Y if possible to preserve layers (X)
                // But for safety, move both
                p2.x -= moveX * 0.5;
                p2.y -= moveY * 0.5;
                p1.x += moveX * 0.5;
                p1.y += moveY * 0.5;
            }
        }
    }

    // 构建顶点映射
    const vertexMap = {};
    diagram.nodes.forEach(node => {
        const pos = positions[node.id];
        if (!pos) {
            console.warn(`⚠️ 节点 ${node.id} 未分配坐标，使用默认值`);
            vertexMap[node.id] = { x: canvasWidth / 2, y: canvasHeight / 2, type: node.type };
        } else {
            vertexMap[node.id] = { x: pos.x, y: pos.y, type: node.type };
        }
    });
    
    // 🔧 检查是否有环路定义
    let loops = diagram.loops || [];

    // 兼容性处理：如果 loops 是数组的数组 (Gemini 返回 [["v1", "v2", "v3"]])
    if (loops.length > 0 && Array.isArray(loops[0])) {
        console.warn('⚠️ Gemini 返回了简化的 loops 格式 (ID数组列表)，自动转换为对象格式');
        loops = loops.map((loopIds, idx) => ({
            vertices: loopIds,
            type: 'penguin_loop', // 默认为企鹅图
            id: `generated_loop_${idx}`
        }));
    }
    // 兼容性处理：如果 loops 是字符串数组 (Gemini 返回 ["v1", "v2", "v3"]) - 虽然少见但以防万一
    else if (loops.length > 0 && typeof loops[0] === 'string') {
        console.warn('⚠️ Gemini 返回了简化的 loops 格式 (单ID数组)，自动转换为对象格式');
        loops = [{
            vertices: [...loops],
            type: 'penguin_loop',
            id: 'generated_loop_0'
        }];
    }

    const hasLoops = loops.length > 0;
    
    // 🔧 如果有环路，将它们转换为可绘制的 loop 形状
    if (hasLoops) {
        loops.forEach((loop, loopIdx) => {
            // 确保环路顶点存在
            const verticesList = loop.vertices || loop.arc || [];
            const loopVertices = verticesList
                .map(vId => vertexMap[vId])
                .filter(v => v); // 过滤掉未找到的顶点

            if (loopVertices.length < 2) {
                console.warn(`环路 ${loop.id} 顶点不足，无法绘制`);
                return;
            }

            console.log(`  ✨ 处理环路顶点位置: ${loop.id || `loop${loopIdx}`}, 类型: ${loop.type}`);

            // 🔧 提取粒子ID（处理对象格式）
            const particleIds = (loop.particles || []).map(p => {
                if (typeof p === 'string') return p;
                if (typeof p === 'object' && p.id) return p.id;
                return 'unknown';
            });
            
            // ⚠️ 禁用 Loop Shape 的直接绘制，因为这会导致与 Edge Shape 重叠 (Double Drawing)
            // 我们只利用 Loop 逻辑来修正顶点位置 (correctedArc)，让 Edge 绘制出正确的形状
            /*
            shapes.push({
                id: Date.now() + 1000 + loopIdx * 10, // 确保ID唯一
                type: 'loop', // 统一使用 'loop' 类型
                loopType: loop.type, // 'penguin_loop' 或其他
                vertices: loopVertices,
                particles: particleIds,
                props: {
                    description: loop.description || ''
                }
            });
            */
            
            // 🔧 如果是企鹅图，重新计算圆弧顶点位置并更新 vertexMap
            // 注意：LayoutEngine 可能已经给出了一个位置，但为了完美的半圆，我们可能需要微调
            // 这里我们信任 LayoutEngine 的拓扑位置，但如果它是 arc_midpoint，我们可能需要根据直径重新计算它的几何位置以确保它是完美的半圆
            if (loop.type === 'penguin_loop' && loopVertices.length === 3) {
                const p1 = loopVertices[0];
                const p2 = loopVertices[1];
                const diameter = Math.hypot(p2.x - p1.x, p2.y - p1.y);
                const radius = diameter / 2;
                
                // 计算理想的圆弧顶点位置
                const correctedArc = {
                    x: (p1.x + p2.x) / 2,
                    y: (p1.y + p2.y) / 2 - radius * 0.85
                };
                
                // 更新 vertexMap 中的圆弧顶点坐标
                const arcVertexId = loop.vertices[2]; // 第三个顶点是圆弧顶点
                if (vertexMap[arcVertexId]) {
                    // 总是修正以保证完美的半圆形状
                    console.log(`🔧 修正企鹅图圆弧顶点 ${arcVertexId}: (${vertexMap[arcVertexId].x.toFixed(1)}, ${vertexMap[arcVertexId].y.toFixed(1)}) -> (${correctedArc.x.toFixed(1)}, ${correctedArc.y.toFixed(1)})`);
                    vertexMap[arcVertexId].x = correctedArc.x;
                    vertexMap[arcVertexId].y = correctedArc.y;
                    loopVertices[2] = correctedArc; // 同时更新 loop 对象中的顶点
                }
            }
        });
    }

    // 🔧 Z-Boson / Photon Emission Angle Fix
    // Ensure neutral bosons emitted from the loop point "outwards" (upwards in this case)
    if (hasLoops) {
        const loopArcVertices = new Set();
        loops.forEach(loop => {
            if (loop.vertices && loop.vertices.length >= 3) {
                // The 3rd vertex is usually the arc top in a penguin loop
                loopArcVertices.add(loop.vertices[2]);
            }
        });

        diagram.edges.forEach(edge => {
            if (['z', 'photon', 'gluon', 'higgs'].includes(edge.particle.id)) {
                if (loopArcVertices.has(edge.source)) {
                    const sourcePos = vertexMap[edge.source];
                    const targetPos = vertexMap[edge.target];
                    
                    if (sourcePos && targetPos) {
                        // Force target to be above source (smaller Y)
                        // And ensure some vertical distance
                        const MIN_VERTICAL_DIST = 60;
                        
                        if (targetPos.y >= sourcePos.y - MIN_VERTICAL_DIST) {
                            console.log(`🔧 Adjusting ${edge.particle.id} emission angle to be upwards`);
                            targetPos.y = sourcePos.y - MIN_VERTICAL_DIST - (Math.random() * 20);
                        }
                    }
                }
            }
        });
    }

    // 🔧 Deduplicate edges to prevent "ghosting" or multiple lines
    const uniqueEdges = [];
    const seenEdges = new Set();
    
    diagram.edges.forEach(edge => {
        // Create a unique key for the edge
        const key = `${edge.source}-${edge.target}-${edge.particle.id}`;
        if (!seenEdges.has(key)) {
            seenEdges.add(key);
            uniqueEdges.push(edge);
        } else {
            console.warn(`⚠️ Duplicate edge detected and removed: ${key}`);
        }
    });

    // 转换线条为 shapes
    uniqueEdges.forEach((edge, idx) => {
        const p1 = vertexMap[edge.source];
        const p2 = vertexMap[edge.target];
        
        if (!p1 || !p2) {
            console.warn(`无法找到顶点 ${edge.source} 或 ${edge.target}`);
            return;
        }
        
        // 构建粒子属性
        const particle = edge.particle;
        
        // 🔧 处理反粒子后缀（如 t_bar -> t + isAnti）
        let particleId = particle.id;
        let isAntiParticle = particle.isAnti || false;
        
        if (particleId.endsWith('_bar')) {
            particleId = particleId.replace('_bar', '');
            isAntiParticle = true;
            console.log(`  🔧 检测到反粒子后缀: ${particle.id} -> ${particleId} (isAnti=true)`);
        }

        // 🔧 处理 q_up_type (GIM 机制中的上型夸克求和)
        if (particleId === 'q_up_type') {
            console.log('  🔧 检测到 q_up_type，映射为顶夸克 (t) 以进行物理验证');
            particleId = 't'; // 默认映射为最重的顶夸克，通常是主导贡献
            // 保持 isAnti 属性不变
        }
        
        const props = {
            category: particle.category,
            particleId: particleId
        };
        
        // ✅ 添加所有 Gemini 提供的属性（保持原始数据）
        if (particle.group === 'lepton' || particle.group === 'quark_u' || particle.group === 'quark_d' || particle.group === 'boson') {
            // 标准格式：直接使用
            props.group = particle.group;
        } else if (particle.group === 'quark' || particle.category === 'fermion') {
            // 🔧 兼容旧格式：如果 group='quark' 或没有 group，自动推断
            const leptons = ['e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau'];
            const quarks_u = ['u', 'c', 't'];
            const quarks_d = ['d', 's', 'b'];
            
            if (leptons.includes(particleId)) {
                props.group = 'lepton';
            } else if (quarks_u.includes(particleId)) {
                props.group = 'quark_u';
                console.log(`  🔧 自动将 ${particle.id} 的 group 从 '${particle.group || 'undefined'}' 修正为 'quark_u'`);
            } else if (quarks_d.includes(particleId)) {
                props.group = 'quark_d';
                console.log(`  🔧 自动将 ${particle.id} 的 group 从 '${particle.group || 'undefined'}' 修正为 'quark_d'`);
            } else {
                console.warn(`未知的费米子 ${particle.id} (mapped to ${particleId})，默认为轻子`);
                props.group = 'lepton';
            }
        }
        if (isAntiParticle) props.isAnti = true;
        if (particle.isVirtual !== undefined) props.isVirtual = particle.isVirtual;
        if (particle.color) {
            props.color = particle.color;
        } else if (particle.category === 'fermion' && (particle.group === 'quark_u' || particle.group === 'quark_d')) {
            // 🔧 如果是夸克但没有指定色荷，默认分配红色
            props.color = particle.isAnti ? 'anti-red' : 'red';
            console.warn(`夸克 ${particle.id} 未指定色荷，默认使用 ${props.color}`);
        }
        if (particle.gluonColor !== undefined) props.gluonColor = particle.gluonColor;
        
        // 🔧 修复反粒子方向和属性
        let finalP1 = p1;
        let finalP2 = p2;
        
        if (particle.category === 'fermion' && particle.isAnti) {
            // 反粒子：箭头应该从右到左（从 p2 到 p1）
            // 交换坐标，使得 p2.x < p1.x，这样 feynman-logic.js 会自动识别为反粒子
            finalP1 = p2;
            finalP2 = p1;
            // ⚠️ 不要在这里设置 props.isAnti，让 feynman-logic.js 根据方向自动判断
        } else if (particle.id === 'w_plus') {
            // 🔧 W+ 玻色子：feynman-logic.js 默认将其视为从右向左流动的粒子 (flowFromV1ToV2 = false)
            // 因此我们需要交换坐标，使得逻辑上的"反向"对应物理上的"正向" (Source -> Target)
            finalP1 = p2;
            finalP2 = p1;
            console.log(`  🔧 检测到 W+ 玻色子，交换坐标以匹配物理流向`);
        } else if (particle.category === 'fermion') {
            // 正粒子：保持原方向（从 p1 到 p2）
            // 确保 p1.x < p2.x
            if (p1.x > p2.x) {
                finalP1 = p2;
                finalP2 = p1;
            }
        }
        
        // 确定线条类型
        let type = 'photon'; // 默认类型
        if (particle.category === 'fermion') {
            type = 'fermion';
        } else if (particle.id === 'photon') {
            type = 'photon';
        } else if (particle.id === 'gluon') {
            type = 'gluon';
        } else if (particle.id === 'w_plus' || particle.id === 'w_minus') {
            type = 'boson_w';
        } else if (particle.id === 'z') {
            type = 'boson_z';
        } else if (particle.id === 'higgs') {
            type = 'higgs';
        }
        
        console.log(`  线条 ${idx}: ${particle.id} -> type=${type}, props.particleId=${props.particleId}`);
        
        shapes.push({
            id: Date.now() + idx * 10,  // 增加 ID 间隔避免冲突
            type: type,
            p1: { x: finalP1.x, y: finalP1.y },
            p2: { x: finalP2.x, y: finalP2.y },
            props: props
        });
    });
    
    // 传播子处理...
    if (diagram.propagators && diagram.propagators.length > 0) {
        console.log(`✨ 检测到 ${diagram.propagators.length} 个传播子`);
        
        diagram.propagators.forEach((prop, propIdx) => {
            const vertex = vertexMap[prop.from];
            if (!vertex) {
                console.warn(`传播子缺少顶点 ${prop.from}`);
                return;
            }
            
            const particle = prop.particle;
            const props = {
                category: particle.category,
                particleId: particle.id,
                isVirtual: true
            };
            
            if (particle.group) props.group = particle.group;
            if (particle.gluonColor !== undefined) props.gluonColor = particle.gluonColor;
            
            // 确定类型
            let type;
            if (particle.id === 'photon') type = 'photon';
            else if (particle.id === 'z') type = 'boson_z';
            else if (particle.id === 'w_plus' || particle.id === 'w_minus') type = 'boson_w';
            else if (particle.id === 'gluon') type = 'gluon';
            else if (particle.id === 'higgs') type = 'higgs';
            else if (particle.category === 'fermion') type = 'fermion';
            else type = 'photon';
            
            console.log(`  ✅ 传播子 ${propIdx + 1}: ${particle.id} (类型: ${type}) 在顶点 ${prop.from} 位置 (${vertex.x}, ${vertex.y})`);
            
            // 🎨 创建一个特殊的"传播子标记"形状
            shapes.push({
                id: Date.now() + 2000 + propIdx * 10,
                type: 'propagator',  // 新类型：传播子标记
                propagatorType: type,
                position: { x: vertex.x, y: vertex.y },
                props: {
                    ...props,
                    isPropagator: true
                }
            });
        });
    }
    
    return {
        shapes,
        explanation: geminiResponse.explanation,
        interactionType: geminiResponse.interaction_type,
        hasLoops: hasLoops
    };
}

/**
 * 主函数：生成费曼图（支持生成所有可能的图表）
 */
async function generateFeynmanDiagram(reactionInput, canvasWidth = 800, canvasHeight = 600, options = {}) {
    try {
        const { generateAll = false } = options;
        
        // 1. 解析反应式
        console.log('📝 解析反应式...');
        const reaction = parseReaction(reactionInput);
        console.log('✅ 解析成功:', reaction);
        
        // 2. 构建初始 prompt
        console.log('🤖 构建 Gemini prompt...');
        let prompt = buildGeminiPrompt(reaction, canvasWidth, canvasHeight);
        
        // 3. 迭代生成与验证循环 (Dimension 5: Iterative Refinement)
        const MAX_RETRIES = 3;
        let lastError = null;
        
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            console.log(`🔄 尝试生成 (第 ${attempt}/${MAX_RETRIES} 次)...`);
            
            if (attempt > 1 && lastError) {
                console.log('⚠️ 上一次生成存在物理错误，正在请求 Gemini 修正...');
                // 将错误信息追加到 prompt
                prompt += `\n\n**PREVIOUS ATTEMPT FAILED**\nYour previous JSON was invalid. \nErrors:\n${lastError}\n\n**PLEASE FIX THE ERRORS AND REGENERATE THE JSON.**`;
            }

            // 调用 Gemini API
            const geminiResponse = await callGeminiAPI(prompt);
            console.log('✅ Gemini 响应:', geminiResponse);
            
            // 验证逻辑
            try {
                // 先尝试转换第一个图表进行验证
                // 注意：convertToShapes 可能会抛出异常（如果 JSON 结构不对）
                const tempResult = convertToShapes(geminiResponse, canvasWidth, canvasHeight);
                
                // 调用物理验证引擎 (如果存在)
                if (window.validateDiagram) {
                    const validation = window.validateDiagram(tempResult.shapes);
                    
                    if (validation.isValid) {
                        console.log('✅ 物理验证通过！');
                        
                        // 验证通过，处理返回结果
                        if (generateAll && geminiResponse.diagrams && geminiResponse.diagrams.length > 1) {
                            // 生成所有图表
                            console.log(`🎨 生成所有 ${geminiResponse.diagrams.length} 个图表...`);
                            const results = [];
                            
                            for (let i = 0; i < geminiResponse.diagrams.length; i++) {
                                const diagram = geminiResponse.diagrams[i];
                                // 为每个图表单独构造响应对象以便转换
                                const singleDiagramResponse = {
                                    valid: geminiResponse.valid,
                                    interaction_type: geminiResponse.interaction_type,
                                    diagrams: [diagram],
                                    explanation: geminiResponse.explanation
                                };
                                
                                const result = convertToShapes(singleDiagramResponse, canvasWidth, canvasHeight);
                                result.diagramName = diagram.name;
                                result.diagramIndex = i;
                                results.push(result);
                            }
                            
                            console.log(`✅ 成功生成 ${results.length} 个图表`);
                            return results;
                            
                        } else {
                            // 只返回第一个图表
                            console.log('🎨 转换为绘图数据...');
                            tempResult.diagramName = geminiResponse.diagrams[0]?.name || 'Main Diagram';
                            console.log('✅ 转换成功，生成了', tempResult.shapes.length, '个形状');
                            return tempResult;
                        }
                    } else {
                        // 验证失败
                        console.warn('⚠️ 物理验证失败:', validation.errors);
                        lastError = validation.errors.join('\n');
                        // 继续下一次循环，Gemini 会收到错误反馈
                        continue;
                    }
                } else {
                    console.warn('⚠️ 找不到 window.validateDiagram，跳过验证');
                    // 如果没有验证引擎，直接返回结果（保持向后兼容）
                    tempResult.diagramName = geminiResponse.diagrams[0]?.name || 'Main Diagram';
                    return tempResult;
                }
                
            } catch (conversionError) {
                console.error('❌ 转换或验证过程出错:', conversionError);
                lastError = conversionError.message;
                // 继续下一次循环
                continue;
            }
        }
        
        throw new Error(`生成失败，已重试 ${MAX_RETRIES} 次。最后一次错误: ${lastError}`);
        
    } catch (error) {
        console.error('❌ 生成费曼图失败:', error);
        throw error;
    }
}

/**
 * 导出函数（供 feynman-logic.js 调用）
 */
window.FeynmanDiagramGenerator = {
    generateFeynmanDiagram,
    parseReaction,
    PARTICLE_SYMBOLS,
    REACTION_DATABASE,
    setAPIKey: (key) => { GEMINI_CONFIG.apiKey = key; }
};

console.log('✅ Gemini Integration 模块已加载');
