// ===== 国际化 =====
let currentLang = localStorage.getItem('language') || 'zh';

const i18n = {
    zh: {
        'back-text': '返回主页',
        'main-title': '粒子物理可视化',
        'lang-viz-text': '中文',
        'force-strong': '强核力',
        'force-em': '电磁力',
        'force-weak': '弱核力',
        'force-gravity': '引力',
        'force-higgs': '希格斯机制',
        'label-strength': '相对强度/性质',
        'label-range': '作用范围',
        'label-particle': '媒介粒子 / 场粒子',
        'label-note': '提示',
        'hint-text': '点击屏幕激发希格斯玻色子',
        
        'strong-name': '强核力',
        'strong-enname': 'Strong Force',
        'strong-desc': '最强的相互作用。夸克通过交换胶子被紧紧束缚。图中红色、绿色、蓝色小球代表夸克(uud)，它们之间的动态连接线代表胶子场（强力胶）。',
        'strong-strength': '1 (最强)',
        'strong-range': '~10⁻¹⁵ m',
        'strong-particle': '胶子 (Gluon)',
        'strong-note': '强核力将质子和中子束缚在原子核中。没有它，宇宙将只有氢原子。',
        
        'em-name': '电磁力',
        'em-enname': 'Electromagnetic',
        'em-desc': '支配电荷的力。图中展示了正电荷（红）和负电荷（蓝）互相吸引，中间流动的光点代表交换的光子（电磁相互作用的媒介）。',
        'em-strength': '≈ 1/137',
        'em-range': '无限 Infinite',
        'em-particle': '光子 (Photon)',
        'em-note': '电磁力负责化学键、光的传播以及我们日常生活中几乎所有的相互作用。',
        
        'weak-name': '弱核力',
        'weak-enname': 'Weak Force',
        'weak-desc': '导致衰变的力。图中展示了β衰变过程：一个中子(n)释放出W玻色子，转变为质子(p)，并产生电子(e)和反中微子(ν)。',
        'weak-strength': '10⁻⁶',
        'weak-range': '10⁻¹⁸ m',
        'weak-particle': 'W⁺, W⁻, Z⁰',
        'weak-note': '弱核力驱动太阳的核聚变，没有它太阳将无法发光，生命也无法存在。',
        
        'gravity-name': '引力',
        'gravity-enname': 'Gravity',
        'gravity-desc': '时空的弯曲。图中绿色网格代表时空，大质量物体运动时会产生"引力波"涟漪，向外扩散，展示了广义相对论的核心概念。',
        'gravity-strength': '10⁻³⁸ (最弱)',
        'gravity-range': '无限 Infinite',
        'gravity-particle': '引力子 (Graviton)',
        'gravity-note': '引力虽然最弱，但因为只有吸引没有排斥，在宏观尺度上主导着宇宙的结构。',
        
        'higgs-name': '希格斯机制',
        'higgs-enname': 'Higgs Mechanism',
        'higgs-desc': '质量的来源。宇宙充满了"希格斯场"（背景点阵）。无质量的光子（上）穿过场不受阻碍。大质量粒子（下）在运动时会扰动希格斯场，产生"阻力"和聚集，这就是质量的本质。',
        'higgs-strength': '赋予质量',
        'higgs-range': '遍布全宇宙',
        'higgs-particle': '希格斯玻色子 (H⁰)',
        'higgs-note': '希格斯场像"宇宙糖浆"，粒子在其中移动越困难，质量就越大。点击画布可激发希格斯玻色子！'
    },
    en: {
        'back-text': 'Back to Home',
        'main-title': 'Particle Physics Visualization',
        'lang-viz-text': 'EN',
        'force-strong': 'Strong',
        'force-em': 'Electromagnetic',
        'force-weak': 'Weak',
        'force-gravity': 'Gravity',
        'force-higgs': 'Higgs',
        'label-strength': 'Relative Strength',
        'label-range': 'Range',
        'label-particle': 'Mediator / Field Particle',
        'label-note': 'Note',
        'hint-text': 'Click to excite Higgs boson',
        
        'strong-name': 'Strong Force',
        'strong-enname': 'Strong Force',
        'strong-desc': 'The strongest interaction. Quarks are tightly bound by exchanging gluons. The red, green, and blue spheres represent quarks (uud), with dynamic connecting lines representing the gluon field (strong glue).',
        'strong-strength': '1 (Strongest)',
        'strong-range': '~10⁻¹⁵ m',
        'strong-particle': 'Gluon',
        'strong-note': 'The strong force binds protons and neutrons in atomic nuclei. Without it, the universe would contain only hydrogen atoms.',
        
        'em-name': 'Electromagnetic Force',
        'em-enname': 'Electromagnetic',
        'em-desc': 'The force governing electric charges. The diagram shows positive (red) and negative (blue) charges attracting each other, with flowing light points representing exchanged photons (mediators of electromagnetic interaction).',
        'em-strength': '≈ 1/137',
        'em-range': 'Infinite',
        'em-particle': 'Photon',
        'em-note': 'Electromagnetic force is responsible for chemical bonds, light propagation, and almost all interactions in daily life.',
        
        'weak-name': 'Weak Force',
        'weak-enname': 'Weak Force',
        'weak-desc': 'The force that causes decay. The diagram shows beta decay: a neutron (n) releases a W boson, transforming into a proton (p), and producing an electron (e) and an antineutrino (ν).',
        'weak-strength': '10⁻⁶',
        'weak-range': '10⁻¹⁸ m',
        'weak-particle': 'W⁺, W⁻, Z⁰',
        'weak-note': 'The weak force drives nuclear fusion in the Sun. Without it, the Sun would not shine and life could not exist.',
        
        'gravity-name': 'Gravity',
        'gravity-enname': 'Gravity',
        'gravity-desc': 'The curvature of spacetime. The green grid represents spacetime, and massive objects in motion create "gravitational wave" ripples that propagate outward, demonstrating the core concept of general relativity.',
        'gravity-strength': '10⁻³⁸ (Weakest)',
        'gravity-range': 'Infinite',
        'gravity-particle': 'Graviton',
        'gravity-note': 'Although gravity is the weakest, it dominates the structure of the universe at macroscopic scales because it only attracts and never repels.',
        
        'higgs-name': 'Higgs Mechanism',
        'higgs-enname': 'Higgs Mechanism',
        'higgs-desc': 'The origin of mass. The universe is filled with the "Higgs field" (background lattice). Massless photons (top) pass through unimpeded. Massive particles (bottom) disturb the Higgs field when moving, creating "drag" and clustering—this is the essence of mass.',
        'higgs-strength': 'Gives Mass',
        'higgs-range': 'Throughout Universe',
        'higgs-particle': 'Higgs Boson (H⁰)',
        'higgs-note': 'The Higgs field acts like "cosmic molasses." The harder a particle moves through it, the greater its mass. Click the canvas to excite a Higgs boson!'
    }
};

function t(key) {
    return i18n[currentLang][key] || key;
}

function updateLanguage() {
    document.getElementById('back-text').textContent = t('back-text');
    document.getElementById('main-title').textContent = t('main-title');
    document.getElementById('lang-viz-text').textContent = t('lang-viz-text');
    document.getElementById('force-strong').textContent = t('force-strong');
    document.getElementById('force-em').textContent = t('force-em');
    document.getElementById('force-weak').textContent = t('force-weak');
    document.getElementById('force-gravity').textContent = t('force-gravity');
    document.getElementById('force-higgs').textContent = t('force-higgs');
    document.getElementById('label-strength').textContent = t('label-strength');
    document.getElementById('label-range').textContent = t('label-range');
    document.getElementById('label-particle').textContent = t('label-particle');
    document.getElementById('label-note').textContent = t('label-note');
    document.getElementById('hint-text').textContent = t('hint-text');
    
    updateUI(currentKey);
}

// ===== 物理数据 =====
const forceData = {
    strong: {
        color: { main: '#ef4444', glow: '#ef4444' }
    },
    em: {
        color: { main: '#3b82f6', glow: '#3b82f6' }
    },
    weak: {
        color: { main: '#eab308', glow: '#eab308' }
    },
    gravity: {
        color: { main: '#22c55e', glow: '#22c55e' }
    },
    higgs: {
        color: { main: '#a855f7', glow: '#d8b4fe' }
    }
};

// ===== 核心变量 =====
let currentKey = 'strong';
let time = 0;
const canvas = document.getElementById('vis-canvas');
const ctx = canvas.getContext('2d', { alpha: false });
let higgsGrid = [];

// ===== UI 元素 =====
const ui = {
    title: document.getElementById('info-title'),
    desc: document.getElementById('info-description'),
    strength: document.getElementById('info-strength'),
    range: document.getElementById('info-range'),
    particle: document.getElementById('info-particle'),
    note: document.getElementById('info-note'),
    floatLabel: document.getElementById('floating-label'),
    hint: document.getElementById('interaction-hint'),
    btns: document.querySelectorAll('.force-btn-desktop')
};

// ===== 初始化与适配 =====
function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    initHiggsGrid(rect.width, rect.height);
}

function initHiggsGrid(w, h) {
    higgsGrid = [];
    const gap = 35;
    for(let x=0; x<w+gap; x+=gap) {
        for(let y=0; y<h+gap; y+=gap) {
            higgsGrid.push({
                baseX: x, baseY: y,
                x: x, y: y,
                vx: 0, vy: 0
            });
        }
    }
}

// ===== 交互逻辑 =====
function updateUI(key) {
    const data = forceData[key];
    
    ui.title.textContent = t(`${key}-name`);
    ui.desc.textContent = t(`${key}-desc`);
    ui.strength.textContent = t(`${key}-strength`);
    ui.range.textContent = t(`${key}-range`);
    ui.particle.textContent = t(`${key}-particle`);
    ui.note.textContent = t(`${key}-note`);
    ui.floatLabel.textContent = t(`${key}-enname`);
    ui.floatLabel.style.color = data.color.glow;
    ui.floatLabel.style.borderColor = data.color.main;

    const btnMap = ['strong', 'em', 'weak', 'gravity', 'higgs'];
    ui.btns.forEach((btn, idx) => {
        const isSelected = btnMap[idx] === key;
        if (isSelected) {
            btn.className = 'force-btn-desktop bg-gray-700 border-2 border-gray-400 text-white font-bold shadow-lg ring-2 ring-gray-500';
        } else {
            btn.className = 'force-btn-desktop bg-gray-800 border border-gray-700 text-gray-400 font-medium hover:bg-gray-750 hover:text-gray-300';
        }
    });

    if (key === 'higgs') ui.hint.classList.remove('hidden');
    else ui.hint.classList.add('hidden');
}

// ===== 点击交互 =====
let userInteractions = [];
canvas.addEventListener('pointerdown', (e) => {
    if (currentKey !== 'higgs') return;
    const rect = canvas.getBoundingClientRect();
    userInteractions.push({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        life: 1.0
    });
});

// ===== 动画循环 =====
function animate(t) {
    const rawTime = t / 1000;
    time = rawTime;
    
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    
    if (currentKey === 'strong') drawStrong(ctx, w, h, time);
    else if (currentKey === 'em') drawEM(ctx, w, h, time);
    else if (currentKey === 'weak') drawWeak(ctx, w, h, time);
    else if (currentKey === 'gravity') drawGravity(ctx, w, h, time);
    else if (currentKey === 'higgs') drawHiggs(ctx, w, h, time);

    ctx.restore();
    requestAnimationFrame(animate);
}

// ===== 绘图函数 =====
function drawStrong(ctx, w, h, t) {
    const cx = w/2, cy = h/2;
    const r = Math.min(w,h)*0.2;
    
    const g = ctx.createRadialGradient(cx,cy, r*0.5, cx,cy, r*1.5);
    g.addColorStop(0, 'rgba(239,68,68,0.2)');
    g.addColorStop(1, 'transparent');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx,cy, r*1.5, 0, Math.PI*2); ctx.fill();

    const qs = [
        {l:'u', c:'#ef4444', a:0},
        {l:'u', c:'#22c55e', a:2.09},
        {l:'d', c:'#3b82f6', a:4.18}
    ].map((q, i) => ({
        ...q,
        x: cx + Math.cos(t*2 + q.a) * (r*0.5 + Math.sin(t*5+i)*10),
        y: cy + Math.sin(t*2 + q.a) * (r*0.5 + Math.sin(t*5+i)*10)
    }));

    ctx.beginPath();
    ctx.moveTo(qs[0].x, qs[0].y);
    ctx.lineTo(qs[1].x, qs[1].y);
    ctx.lineTo(qs[2].x, qs[2].y);
    ctx.closePath();
    ctx.lineWidth = 8;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.stroke();
    
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.setLineDash([6, 12]);
    ctx.lineDashOffset = -t*30;
    ctx.stroke();
    ctx.setLineDash([]);

    qs.forEach(q => {
        ctx.shadowBlur = 20; ctx.shadowColor = q.c;
        ctx.fillStyle = q.c;
        ctx.beginPath(); ctx.arc(q.x, q.y, r*0.28, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'white';
        ctx.font = 'bold 22px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(q.l, q.x, q.y);
    });
}

function drawEM(ctx, w, h, t) {
    const cx = w/2, cy = h/2;
    const dist = 160;
    const p1 = {x: cx-dist/2, y: cy, c:'#ef4444', s:'+'};
    const p2 = {x: cx+dist/2, y: cy, c:'#3b82f6', s:'-'};

    for(let i=0; i<10; i++) {
        const offset = (i-4.5) * 18;
        const flow = (t * 2 + i*0.5) % 1;
        
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.bezierCurveTo(cx, cy-offset*5, cx, cy-offset*5, p2.x, p2.y);
        ctx.strokeStyle = `rgba(100,150,255,${0.2 + Math.abs(offset)/120})`;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        if(i%2===0) {
            const t_val = flow;
            const invT = 1 - t_val;
            const bx = invT*invT*p1.x + 2*invT*t_val*cx + t_val*t_val*p2.x;
            const by = invT*invT*p1.y + 2*invT*t_val*(cy-offset*5) + t_val*t_val*p2.y;
            
            ctx.fillStyle = 'white';
            ctx.shadowBlur = 8; ctx.shadowColor = 'white';
            ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI*2); ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    [p1, p2].forEach(p => {
        ctx.shadowBlur = 25; ctx.shadowColor = p.c;
        ctx.fillStyle = p.c;
        ctx.beginPath(); ctx.arc(p.x, p.y, 30, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'white';
        ctx.font = 'bold 28px monospace';
        ctx.textAlign = 'center'; 
        ctx.textBaseline = 'middle';
        ctx.fillText(p.s, p.x, p.y+2);
    });
}

function drawWeak(ctx, w, h, t) {
    const cx=w/2, cy=h/2;
    const cycle = t%5;
    
    ctx.font='bold 18px sans-serif';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    
    if(cycle < 2) {
        const shake = Math.sin(t*30)*2;
        ctx.fillStyle = '#888';
        ctx.beginPath(); ctx.arc(cx+shake, cy, 35, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText('n', cx+shake, cy);
        ctx.fillStyle='#aaa';
        ctx.font='14px sans-serif';
        ctx.fillText(currentLang === 'zh' ? '中子衰变中...' : 'Neutron decay...', cx, cy+60);
    } else if (cycle < 2.5) {
        const p = (cycle-2)/0.5;
        ctx.fillStyle = '#ef4444';
        ctx.beginPath(); ctx.arc(cx, cy, 35, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText('p', cx, cy);
        
        const wx = cx + p*100, wy = cy - p*50;
        ctx.shadowBlur=18; ctx.shadowColor='#eab308';
        ctx.fillStyle = '#eab308';
        ctx.beginPath(); ctx.arc(wx, wy, 24, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur=0;
        ctx.fillStyle = 'black';
        ctx.fillText('W⁻', wx, wy);
    } else {
        const p = (cycle-2.5)/2.5;
        ctx.fillStyle = '#ef4444'; 
        ctx.beginPath(); ctx.arc(cx, cy, 35, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText('p', cx, cy);

        const ex = (cx+100) + p*250, ey = (cy-50) - p*120;
        ctx.fillStyle = 'cyan';
        ctx.beginPath(); ctx.arc(ex, ey, 12, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'black';
        ctx.font='bold 14px sans-serif';
        ctx.fillText('e⁻', ex, ey);

        const nx = (cx+100) + p*180, ny = (cy-50) + p*180;
        ctx.fillStyle = 'white';
        ctx.beginPath(); ctx.arc(nx, ny, 6, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'black';
        ctx.font='bold 16px sans-serif';
        ctx.fillText('ν', nx+12, ny);
    }
}

function drawGravity(ctx, w, h, t) {
    const cx = w/2, cy = h/2;
    const planetX = cx + Math.cos(t)*120;
    const planetY = cy + Math.sin(t)*100;

    ctx.lineWidth = 1.5;
    for(let x=0; x<=w; x+=45) {
        ctx.beginPath();
        for(let y=0; y<=h; y+=12) {
            const dx = x - planetX, dy = y - planetY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            const wave = Math.sin(dist*0.08 - t*8) * Math.max(0, 22 - dist*0.05);
            const pull = Math.max(0, (1 - dist/160) * 35);

            const offX = (dx/ (dist+1)) * (wave*0.5 - pull);
            const offY = (dy/ (dist+1)) * (wave*0.5 - pull);

            ctx.strokeStyle = `rgba(34, 197, 94, ${0.12 + Math.abs(wave)/50})`;
            if(y===0) ctx.moveTo(x+offX, y+offY);
            else ctx.lineTo(x+offX, y+offY);
        }
        ctx.stroke();
    }

    ctx.shadowBlur = 25; ctx.shadowColor = '#22c55e';
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath(); ctx.arc(planetX, planetY, 18, 0, Math.PI*2); ctx.fill();
    ctx.shadowBlur = 0;
}

function drawHiggs(ctx, w, h, t) {
    const photonX = (t * 600) % (w + 200) - 100;
    const photonY = h * 0.3;
    
    const heavyX = (t * 150) % (w + 200) - 100;
    const heavyY = h * 0.65;

    ctx.fillStyle = '#a855f7';
    
    higgsGrid.forEach(pt => {
        const dx = pt.baseX - heavyX;
        const dy = pt.baseY - heavyY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        let targetX = pt.baseX;
        let targetY = pt.baseY;

        if (dist < 140) {
            const pull = (1 - dist/140) * 40;
            targetX -= (dx/dist) * pull;
            targetY -= (dy/dist) * pull;
        }

        userInteractions.forEach(int => {
            const idx = pt.baseX - int.x;
            const idy = pt.baseY - int.y;
            const idist = Math.sqrt(idx*idx + idy*idy);
            const wave = Math.sin(idist*0.1 - (1-int.life)*10) * int.life * 18;
            targetY += wave;
        });

        pt.vx += (targetX - pt.x) * 0.1;
        pt.vy += (targetY - pt.y) * 0.1;
        pt.vx *= 0.8;
        pt.vy *= 0.8;
        pt.x += pt.vx;
        pt.y += pt.vy;

        const displacement = Math.abs(pt.x - pt.baseX) + Math.abs(pt.y - pt.baseY);
        const alpha = 0.15 + Math.min(0.8, displacement/10);
        ctx.globalAlpha = alpha;
        ctx.beginPath(); ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI*2); ctx.fill();
    });
    ctx.globalAlpha = 1.0;

    ctx.shadowBlur = 18; ctx.shadowColor = '#fff';
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(photonX, photonY, 7, 0, Math.PI*2); ctx.fill();
    ctx.shadowBlur = 0;
    
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(photonX, photonY); ctx.lineTo(photonX-60, photonY); ctx.stroke();
    
    ctx.fillStyle = '#ddd'; ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    const photonLabel = currentLang === 'zh' ? '光子 (无质量/不与场作用)' : 'Photon (massless/no interaction)';
    ctx.fillText(photonLabel, photonX, photonY - 20);

    ctx.shadowBlur = 24; ctx.shadowColor = '#d8b4fe';
    ctx.fillStyle = '#d8b4fe';
    ctx.beginPath(); ctx.arc(heavyX, heavyY, 28, 0, Math.PI*2); ctx.fill();
    ctx.shadowBlur = 0;

    const glowSize = 34 + Math.sin(t*10)*6;
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(heavyX, heavyY, glowSize, 0, Math.PI*2); ctx.stroke();

    ctx.fillStyle = '#ddd';
    const heavyLabel = currentLang === 'zh' ? '重粒子 (与场耦合/获得质量)' : 'Heavy particle (coupled/gains mass)';
    ctx.fillText(heavyLabel, heavyX, heavyY + 50);

    userInteractions = userInteractions.filter(int => int.life > 0);
    userInteractions.forEach(int => {
        int.life -= 0.02;
        ctx.font = 'bold 24px serif';
        ctx.fillStyle = `rgba(255, 255, 255, ${int.life})`;
        ctx.fillText("H⁰", int.x, int.y - (1-int.life)*60);
        
        ctx.strokeStyle = `rgba(216, 180, 254, ${int.life})`;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(int.x, int.y, (1-int.life)*70, 0, Math.PI*2); ctx.stroke();
    });
}

// ===== 样式定义 =====
const style = document.createElement('style');
style.textContent = `
    .force-btn-desktop {
        padding: 0.75rem;
        border-radius: 0.75rem;
        font-size: 0.875rem;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        cursor: pointer;
    }
    .force-btn-desktop:hover {
        transform: scale(1.02);
    }
    .force-btn-desktop:active {
        transform: scale(0.98);
    }
`;
document.head.appendChild(style);

// ===== 绑定按钮事件 =====
document.querySelectorAll('.force-btn-desktop').forEach((btn, idx) => {
    const keys = ['strong', 'em', 'weak', 'gravity', 'higgs'];
    btn.onclick = () => {
        currentKey = keys[idx];
        updateUI(currentKey);
    };
});

// ===== 语言切换 =====
document.getElementById('btn-lang-viz').onclick = () => {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    localStorage.setItem('language', currentLang);
    updateLanguage();
};

// ===== 返回按钮 =====
document.getElementById('btn-back').onclick = () => {
    window.location.href = 'start.html';
};

// ===== 启动 =====
window.addEventListener('resize', resize);
resize();
updateLanguage();
requestAnimationFrame(animate);