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
    // 标准化输入 - 移除多余空格
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

**CRITICAL RULES:**
1.  **ONLY generate LOWEST-ORDER (tree-level) diagrams** - NO loop diagrams, NO radiative corrections
2.  **For flow diagrams: Minimize additional gluon exchanges** - Since we draw the lowest-degree Feynman diagrams, avoid drawing extra gluons to keep diagrams simple. However, strong interactions are still valid - this is about diagram simplicity, not physics constraints.
3.  **Generate ALL possible tree-level channels:**
    *   For e⁻e⁺ → μ⁻μ⁺: Generate s-channel with virtual photon (γ*) AND Z⁰ (two separate diagrams)
    *   For e⁻e⁺ → e⁻e⁺ (Bhabha): Generate s-channel AND t-channel (two separate diagrams)
    *   For quark weak decays: Generate the tree-level W± exchange diagram ONLY
    *   For FCNC processes: If tree-level is FORBIDDEN, generate penguin loop diagram (as the lowest physical order)
4.  **Put each diagram in the "diagrams" array** - Each channel is a separate diagram object

**DIAGRAM TYPE SELECTION:**
-   **DEFAULT: Tree-level diagrams ONLY** (e⁻e⁺→μ⁻μ⁺, quark weak decay, etc.)
-   **EXCEPTION: For FCNC processes ONLY** (b→s transitions with γ/Z, forbidden at tree level): Use penguin loop diagram as the "lowest physical order"
-   **For simplicity in lowest-order diagrams: Avoid additional gluon radiative corrections** - Focus on the dominant process, but strong interactions are still physically valid

**EXAMPLES OF LOWEST-ORDER DIAGRAMS:**
-   e⁻ + e⁺ → μ⁻ + μ⁺: Two tree-level diagrams (s-channel via γ*, s-channel via Z⁰)
-   e⁻ + e⁺ → e⁻ + e⁺: Two tree-level diagrams (s-channel via γ*, t-channel via γ)
-   u → d + W⁺ (quark weak decay): One tree-level diagram (gluon corrections omitted for simplicity)
-   b → s + γ (FCNC): One loop diagram (penguin with W boson loop, quarks on horizontal line) - exception because tree-level is forbidden

**CRITICAL REQUIREMENTS:**
1.  ALL initial state particles MUST start from x=80 (left boundary)
2.  ALL final state particles MUST end at x=${canvasWidth - 80} (right boundary)
3.  Create intermediate vertices in the middle region (x around ${canvasWidth / 2})
4.  **Virtual particle lines MUST have visible length (at least 100px separation between vertices)**
5.  EVERY line must connect two vertices - no dangling lines
6.  For loop diagrams, define the loop structure explicitly

**Canvas dimensions:** ${canvasWidth} × ${canvasHeight}

**Physical Rules (Tree-Level Only):**
-   Photons (γ) only couple to charged particles
-   **For lowest-order diagrams: Minimize additional gluon exchanges for simplicity** - Gluons are valid for strong interactions, but we focus on the simplest lowest-degree diagrams
-   W± bosons can change quark flavor (u and d, c and s, t and b) at tree level
-   Z0 bosons couple to all fermions but do NOT change flavor
-   Conserve charge, lepton number, baryon number at each vertex
-   **FCNC processes (b→s, s→d with γ/Z) require loop diagrams** - Tree level is forbidden by SM
-   Virtual particles connect internal vertices only

**PENGUIN DIAGRAM STRUCTURE (Semi-Circle Shape):**
**CRITICAL: The penguin diagram MUST be a semi-circle.**
-   The ARC of the semi-circle is the W boson.
-   The DIAMETER of the semi-circle is the intermediate quark line.
-   The photon/Z boson radiates from the W boson arc.

**PENGUIN DIAGRAM JSON STRUCTURE:**
-   The "loops" array should contain ONE object with \`type: "penguin_loop"\`.
-   The \`vertices\` property in the loop object MUST contain exactly 3 vertices: \`[start_of_diameter, end_of_diameter, arc_midpoint]\`.
-   The \`particles\` property should list: \`[intermediate_quark_id, w_boson_id]\`.
-   **CRITICAL**: Use standard particle IDs only: "w_plus" or "w_minus" (NOT "w_boson").
-   **For anti-quarks, use standard IDs with isAnti flag**: e.g., {"id": "t", "isAnti": true} NOT "t_bar".

**CORRECT PENGUIN JSON EXAMPLE:**
\\\`\\\`\\\`json
{
  "valid": true,
  "interaction_type": "weak_loop",
  "diagrams": [
    {
      "name": "penguin-diagram",
      "vertices": [
        {"id": "v1", "position": {"x": 80, "y": 300}},
        {"id": "v2", "position": {"x": 300, "y": 300}},
        {"id": "v3", "position": {"x": 700, "y": 300}},
        {"id": "v4", "position": {"x": 500, "y": 150}},
        {"id": "v6", "position": {"x": 900, "y": 300}},
        {"id": "v7", "position": {"x": 1100, "y": 200}},
        {"id": "v8", "position": {"x": 1100, "y": 400}}
      ],
      "lines": [
        {"from": "v1", "to": "v2", "particle": {"id": "b", "category": "fermion", "group": "quark_d", "color": "red"}},
        {"from": "v3", "to": "v6", "particle": {"id": "s", "category": "fermion", "group": "quark_d", "color": "red"}},
        {"from": "v4", "to": "v6", "particle": {"id": "photon", "category": "boson", "isVirtual": true}},
        {"from": "v6", "to": "v7", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": false}},
        {"from": "v6", "to": "v8", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": true}}
      ],
      "loops": [
        {
          "id": "loop1",
          "type": "penguin_loop",
          "vertices": ["v2", "v3", "v4"],
          "particles": ["u", "w_minus"],
          "description": "Penguin loop: b->s transition. Diameter (v2-v3) is intermediate u-quark. Arc is W-minus boson (as two lines). Photon radiates from W arc midpoint."
        }
      ]
    }
  ],
  "explanation": "Penguin diagram: b→s transition via a semi-circle loop. The diameter is the intermediate up-type quark and the arc is the W boson. A photon is emitted from the W boson arc."
}
\\\`\\\`\\\`

**TREE-LEVEL DIAGRAM STRUCTURE (for simple processes):**
\\\`\\\`\\\`json
{
  "valid": true,
  "interaction_type": "electromagnetic",
  "diagrams": [
    {
      "name": "s-channel-photon",
      "vertices": [
        {"id": "v1", "position": {"x": 80, "y": ${canvasHeight / 2 - 100}}},
        {"id": "v2", "position": {"x": 80, "y": ${canvasHeight / 2 + 100}}},
        {"id": "v3", "position": {"x": ${canvasWidth / 2 - 100}, "y": ${canvasHeight / 2}}},
        {"id": "v4", "position": {"x": ${canvasWidth / 2 + 100}, "y": ${canvasHeight / 2}}},
        {"id": "v5", "position": {"x": ${canvasWidth - 80}, "y": ${canvasHeight / 2 - 100}}},
        {"id": "v6", "position": {"x": ${canvasWidth - 80}, "y": ${canvasHeight / 2 + 100}}}
      ],
      "lines": [
        {"from": "v1", "to": "v3", "particle": {"id": "e", "category": "fermion", "group": "lepton", "isAnti": false}},
        {"from": "v2", "to": "v3", "particle": {"id": "e", "category": "fermion", "group": "lepton", "isAnti": true}},
        {"from": "v3", "to": "v4", "particle": {"id": "photon", "category": "boson", "isVirtual": true}},
        {"from": "v4", "to": "v5", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": false}},
        {"from": "v4", "to": "v6", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": true}}
      ]
    },
    {
      "name": "s-channel-z-boson",
      "vertices": [
        {"id": "v1", "position": {"x": 80, "y": ${canvasHeight / 2 - 100}}},
        {"id": "v2", "position": {"x": 80, "y": ${canvasHeight / 2 + 100}}},
        {"id": "v3", "position": {"x": ${canvasWidth / 2 - 100}, "y": ${canvasHeight / 2}}},
        {"id": "v4", "position": {"x": ${canvasWidth / 2 + 100}, "y": ${canvasHeight / 2}}},
        {"id": "v5", "position": {"x": ${canvasWidth - 80}, "y": ${canvasHeight / 2 - 100}}},
        {"id": "v6", "position": {"x": ${canvasWidth - 80}, "y": ${canvasHeight / 2 + 100}}}
      ],
      "lines": [
        {"from": "v1", "to": "v3", "particle": {"id": "e", "category": "fermion", "group": "lepton", "isAnti": false}},
        {"from": "v2", "to": "v3", "particle": {"id": "e", "category": "fermion", "group": "lepton", "isAnti": true}},
        {"from": "v3", "to": "v4", "particle": {"id": "z", "category": "boson", "isVirtual": true}},
        {"from": "v4", "to": "v5", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": false}},
        {"from": "v4", "to": "v6", "particle": {"id": "mu", "category": "fermion", "group": "lepton", "isAnti": true}}
      ]
    }
  ],
  "explanation": "s-channel diagrams: e⁻e⁺ can annihilate via virtual photon γ* or Z⁰ boson, producing μ⁻μ⁺ pair"
}
\\\`\\\`\\\`

**CRITICAL - Lowest Order Diagrams Only:** 
-   **Generate ALL tree-level channels** (e.g., both γ* and Z⁰ for e⁻e⁺→μ⁻μ⁺, both s-channel and t-channel for scattering)
-   **Minimize additional gluon radiative corrections** - For simplicity in lowest-order flow diagrams, avoid extra gluons (though they are physically valid for strong interactions)
-   **NO Higgs exchange** unless explicitly requested (Higgs coupling is very weak)
-   Each channel in separate object in "diagrams" array
-   Virtual particle line MUST have at least 100px horizontal separation (e.g., v3 at x=620, v4 at x=820)
-   Initial particles (e⁻, e⁺) start at x=80
-   Final particles (μ⁻, μ⁺) end at x=${canvasWidth - 80}

**IMPORTANT RULES:** 
-   Particle IDs: e, mu, tau, nu_e, nu_mu, nu_tau, u, d, c, s, t, b, photon, w_plus, w_minus, z, higgs, gluon
-   **For lowest-order flow diagrams: Minimize gluon additions** - Gluons are valid for strong interactions, but we focus on simplest diagrams
-   For fermions, set "isAnti": true for antiparticles
-   **For QUARKS, MUST include "color": "red"/"green"/"blue"**
-   **For ANTIQUARKS, MUST include "color": "anti-red"/"anti-green"/"anti-blue"**
-   **For virtual particles in loops, set "isVirtual": true**
-   **FCNC processes (b→s, s→d with Z/γ) use loop diagrams with "loops" array** (exception: tree level forbidden)
-   **For PENGUIN loops, \`type\` MUST be \`penguin_loop\` and \`vertices\` MUST be \`[diameter_start, diameter_end, arc_midpoint]\`**
-   Generate the complete JSON now for: ${initialParticles} → ${finalParticles}`;

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
    // TODO: 未来可以添加 UI 让用户选择要显示哪个图
    const diagram = geminiResponse.diagrams[0];
    
    if (geminiResponse.diagrams.length > 1) {
        console.log(`✨ 可选图表: ${geminiResponse.diagrams.map(d => d.name).join(', ')}`);
        console.log(`📌 当前显示: ${diagram.name}`);
    }
    
    // 检查图表是否有必要的数据
    if (!diagram.vertices || !diagram.lines) {
        throw new Error('费曼图缺少顶点或线条数据');
    }
    
    // 构建顶点映射
    const vertexMap = {};
    diagram.vertices.forEach(v => {
        vertexMap[v.id] = {
            x: v.position.x,
            y: v.position.y,
            type: v.type || 'normal'  // 标记顶点类型：external, internal, loop
        };
    });
    
    // 🔧 检查是否有环路定义
    const hasLoops = diagram.loops && diagram.loops.length > 0;
    
    // 🔧 如果有环路，将它们转换为可绘制的 loop 形状
    if (hasLoops) {
        diagram.loops.forEach((loop, loopIdx) => {
            // 确保环路顶点存在
            const loopVertices = loop.vertices
                .map(vId => vertexMap[vId])
                .filter(v => v); // 过滤掉未找到的顶点

            if (loopVertices.length < 2) {
                console.warn(`环路 ${loop.id} 顶点不足，无法绘制`);
                return;
            }

            console.log(`  ✨ 创建环路形状: ${loop.id || `loop${loopIdx}`}, 类型: ${loop.type}`);

            // 🔧 提取粒子ID（处理对象格式）
            const particleIds = (loop.particles || []).map(p => {
                if (typeof p === 'string') return p;
                if (typeof p === 'object' && p.id) return p.id;
                return 'unknown';
            });
            
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
            
            // 🔧 如果是企鹅图，重新计算圆弧顶点位置并更新 vertexMap
            if (loop.type === 'penguin_loop' && loopVertices.length === 3) {
                const p1 = loopVertices[0];
                const p2 = loopVertices[1];
                const diameter = Math.hypot(p2.x - p1.x, p2.y - p1.y);
                const radius = diameter / 2;
                const correctedArc = {
                    x: (p1.x + p2.x) / 2,
                    y: (p1.y + p2.y) / 2 - radius * 0.85
                };
                
                // 更新 vertexMap 中的圆弧顶点坐标
                const arcVertexId = loop.vertices[2]; // 第三个顶点是圆弧顶点
                if (vertexMap[arcVertexId]) {
                    console.log(`🔧 修正企鹅图圆弧顶点 ${arcVertexId}: (${vertexMap[arcVertexId].x}, ${vertexMap[arcVertexId].y}) -> (${correctedArc.x}, ${correctedArc.y})`);
                    vertexMap[arcVertexId].x = correctedArc.x;
                    vertexMap[arcVertexId].y = correctedArc.y;
                    loopVertices[2] = correctedArc; // 同时更新 loop 对象中的顶点
                }
            }
        });
    }

    // 转换线条为 shapes
    diagram.lines.forEach((line, idx) => {
        const p1 = vertexMap[line.from];
        const p2 = vertexMap[line.to];
        
        if (!p1 || !p2) {
            console.warn(`无法找到顶点 ${line.from} 或 ${line.to}`);
            return;
        }
        
        // 构建粒子属性
        const particle = line.particle;
        
        // 🔧 处理反粒子后缀（如 t_bar -> t + isAnti）
        let particleId = particle.id;
        let isAntiParticle = particle.isAnti || false;
        
        if (particleId.endsWith('_bar')) {
            particleId = particleId.replace('_bar', '');
            isAntiParticle = true;
            console.log(`  🔧 检测到反粒子后缀: ${particle.id} -> ${particleId} (isAnti=true)`);
        }
        
        const props = {
            category: particle.category,
            particleId: particleId
        };
        
        // ✅ 添加所有 Gemini 提供的属性（保持原始数据）
        if (particle.group === 'lepton' || particle.group === 'quark_u' || particle.group === 'quark_d') {
            // 标准格式：直接使用
            props.group = particle.group;
        } else if (particle.group === 'quark' || particle.category === 'fermion') {
            // 🔧 兼容旧格式：如果 group='quark' 或没有 group，自动推断
            const leptons = ['e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau'];
            const quarks_u = ['u', 'c', 't'];
            const quarks_d = ['d', 's', 'b'];
            
            if (leptons.includes(particle.id)) {
                props.group = 'lepton';
            } else if (quarks_u.includes(particle.id)) {
                props.group = 'quark_u';
                console.log(`  🔧 自动将 ${particle.id} 的 group 从 '${particle.group || 'undefined'}' 修正为 'quark_u'`);
            } else if (quarks_d.includes(particle.id)) {
                props.group = 'quark_d';
                console.log(`  🔧 自动将 ${particle.id} 的 group 从 '${particle.group || 'undefined'}' 修正为 'quark_d'`);
            } else {
                console.warn(`未知的费米子 ${particle.id}，默认为轻子`);
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
        
        // 2. 构建 prompt
        console.log('🤖 构建 Gemini prompt...');
        const prompt = buildGeminiPrompt(reaction, canvasWidth, canvasHeight);
        
        // 3. 调用 Gemini API
        console.log('🌐 调用 Gemini 2.5 Pro API...');
        const geminiResponse = await callGeminiAPI(prompt);
        console.log('✅ Gemini 响应:', geminiResponse);
        
        // 4. 根据选项生成单个或多个图表
        if (generateAll && geminiResponse.diagrams && geminiResponse.diagrams.length > 1) {
            // 生成所有图表，返回数组
            console.log(`🎨 生成所有 ${geminiResponse.diagrams.length} 个图表...`);
            const results = [];
            
            for (let i = 0; i < geminiResponse.diagrams.length; i++) {
                const diagram = geminiResponse.diagrams[i];
                console.log(`  处理第 ${i + 1} 个图表: ${diagram.name}`);
                
                // 创建临时响应对象（只包含当前图表）
                const singleDiagramResponse = {
                    valid: geminiResponse.valid,
                    interaction_type: geminiResponse.interaction_type,
                    diagrams: [diagram],
                    explanation: geminiResponse.explanation
                };
                
                const result = convertToShapes(singleDiagramResponse, canvasWidth, canvasHeight);
                result.diagramName = diagram.name; // 添加图表名称
                result.diagramIndex = i;
                results.push(result);
            }
            
            console.log(`✅ 成功生成 ${results.length} 个图表`);
            return results;
            
        } else {
            // 只生成第一个图表
            console.log('🎨 转换为绘图数据...');
            const result = convertToShapes(geminiResponse, canvasWidth, canvasHeight);
            result.diagramName = geminiResponse.diagrams[0]?.name || 'Main Diagram';
            console.log('✅ 转换成功，生成了', result.shapes.length, '个形状');
            return result;
        }
        
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
