const translations = {
    en: {
        subtitle: "Feynman Diagram Drawing + Force Visualization System",
        suite_title: "Particle Physics Visualization Suite",
        feynman_title: "Feynman Diagram Drawing",
        feynman_subtitle: "Feynman Diagram Editor",
        feynman_feat1: "Draw particles like fermions, bosons, gluons, etc.",
        feynman_feat2: "Auto-validate conservation laws (charge, lepton/baryon numbers, color charge)",
        feynman_feat3: "Smart endpoint snapping, multi-select, undo/redo",
        feynman_feat4: "Bilingual interface (Chinese/English)",
        feynman_cta: "Start Drawing →",
        feynman_badge: "Recommended",
        force_title: "Force Visualization",
        force_subtitle: "Force Visualization",
        force_feat1: "Animate the 5 fundamental forces (Strong, Weak, EM, Gravity, Higgs)",
        force_feat2: "Real-time interactive particle simulation",
        force_feat3: "Detailed physical explanations and parameter displays",
        force_feat4: "Desktop-optimized for a large-screen experience",
        force_cta: "Explore Forces →",
        force_badge: "Interactive",
        info_frontend: "Pure Frontend • No Installation",
        info_opensource: "Open Source • Customizable",
        info_fullscreen: "Press F11 for a better fullscreen experience",
    },
    zh: {
        subtitle: "费曼图绘制 + 力可视化系统",
        suite_title: "粒子物理可视化套件",
        feynman_title: "费曼图绘制",
        feynman_subtitle: "Feynman Diagram Editor",
        feynman_feat1: "绘制费米子、玻色子、胶子等粒子",
        feynman_feat2: "自动验证守恒定律 (电荷、轻子数、重子数、色荷)",
        feynman_feat3: "智能端点吸附、多选、撤销功能",
        feynman_feat4: "中英文双语界面",
        feynman_cta: "开始绘制 →",
        feynman_badge: "推荐",
        force_title: "力可视化",
        force_subtitle: "Force Visualization",
        force_feat1: "动画演示五大基本力 (强、弱、电磁、引力、希格斯)",
        force_feat2: "实时交互式粒子模拟",
        force_feat3: "详细物理解释和参数展示",
        force_feat4: "桌面优化、大屏体验",
        force_cta: "探索基本力 →",
        force_badge: "互动",
        info_frontend: "纯前端 • 无需安装",
        info_opensource: "开源 • 可定制",
        info_fullscreen: "按 F11 全屏体验更佳",
    }
};

document.addEventListener('DOMContentLoaded', () => {
    const langToggle = document.getElementById('language-toggle');
    const elementsToTranslate = document.querySelectorAll('[data-lang-key]');

    function setLanguage(lang) {
        elementsToTranslate.forEach(el => {
            const key = el.getAttribute('data-lang-key');
            if (translations[lang] && translations[lang][key]) {
                el.innerHTML = translations[lang][key];
            }
        });
        localStorage.setItem('language', lang);
        if (lang === 'en') {
            langToggle.checked = true;
        } else {
            langToggle.checked = false;
        }
    }

    langToggle.addEventListener('change', (event) => {
        if (event.target.checked) {
            setLanguage('en');
        } else {
            setLanguage('zh');
        }
    });

    // Set initial language based on localStorage or browser language
    const savedLang = localStorage.getItem('language');
    const browserLang = navigator.language.startsWith('zh') ? 'zh' : 'en';
    const initialLang = savedLang || browserLang;
    setLanguage(initialLang);
});
