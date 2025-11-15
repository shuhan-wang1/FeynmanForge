/**
 * Canvas Manager UI - 画布管理界面逻辑
 */

(function() {
    let canvasManager = null;
    
    // 初始化
    function initCanvasManagerUI() {
        canvasManager = new CanvasManager();
        
        // 尝试从 LocalStorage 加载
        if (!canvasManager.loadFromLocalStorage()) {
            console.log('📝 创建默认画布');
        }
        
        // 绑定事件
        bindEvents();
        
        // 更新UI
        updateUI();
        
        console.log('✅ 画布管理系统初始化完成');
    }
    
    // 绑定事件
    function bindEvents() {
        // 新建画布按钮
        document.getElementById('btn-new-canvas')?.addEventListener('click', () => {
            const name = prompt('请输入新画布名称:', `Diagram ${canvasManager.canvases.length + 1}`);
            if (name) {
                canvasManager.createNewCanvas(name);
                // 清空当前画布
                if (window.clearAllShapes) {
                    window.clearAllShapes();
                }
                showToast(`✅ 已创建新画布: ${name}`);
            }
        });
        
        // 画布列表按钮
        document.getElementById('btn-canvas-list')?.addEventListener('click', toggleCanvasMenu);
        
        // 关闭菜单
        document.getElementById('btn-close-canvas-menu')?.addEventListener('click', hideCanvasMenu);
        
        // 导出画布
        document.getElementById('btn-export-canvases')?.addEventListener('click', exportAllCanvases);
        
        // 导入画布
        document.getElementById('btn-import-canvases')?.addEventListener('click', importCanvases);
        
        // 监听画布更新事件
        window.addEventListener('canvasManagerUpdate', (e) => {
            updateUI();
            updateCanvasListMenu();
        });
        
        // 监听 shapes 变化，自动保存到当前画布
        // 这个会在 feynman-logic.js 中触发
        window.addEventListener('shapesUpdated', (e) => {
            if (canvasManager && e.detail && e.detail.shapes) {
                canvasManager.updateCurrentCanvas(e.detail.shapes);
            }
        });
    }
    
    // 更新UI显示
    function updateUI() {
        const currentCanvas = canvasManager.getCurrentCanvas();
        if (currentCanvas) {
            document.getElementById('current-canvas-name').textContent = currentCanvas.name;
        }
    }
    
    // 切换画布菜单
    function toggleCanvasMenu() {
        const menu = document.getElementById('canvas-list-menu');
        if (menu.classList.contains('hidden')) {
            showCanvasMenu();
        } else {
            hideCanvasMenu();
        }
    }
    
    // 显示画布菜单
    function showCanvasMenu() {
        const menu = document.getElementById('canvas-list-menu');
        const btn = document.getElementById('btn-canvas-list');
        
        // 定位菜单
        const rect = btn.getBoundingClientRect();
        menu.style.left = `${rect.left}px`;
        menu.style.top = `${rect.bottom + 8}px`;
        
        menu.classList.remove('hidden');
        updateCanvasListMenu();
    }
    
    // 隐藏画布菜单
    function hideCanvasMenu() {
        document.getElementById('canvas-list-menu').classList.add('hidden');
    }
    
    // 更新画布列表内容
    function updateCanvasListMenu() {
        const container = document.getElementById('canvas-list-content');
        const canvases = canvasManager.getAllCanvases();
        
        container.innerHTML = canvases.map((canvas, index) => `
            <div class="p-2 rounded ${canvas.isCurrent ? 'bg-cyan-600/20 border border-cyan-600/50' : 'bg-slate-700/30 hover:bg-slate-700/50'} mb-2 transition-all group">
                <div class="flex items-center justify-between">
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2">
                            ${canvas.isCurrent ? '<i data-lucide="check-circle" class="w-4 h-4 text-cyan-400 flex-shrink-0"></i>' : '<i data-lucide="file" class="w-4 h-4 text-slate-400 flex-shrink-0"></i>'}
                            <span class="text-sm font-medium text-slate-200 truncate">${canvas.name}</span>
                        </div>
                        <div class="text-xs text-slate-500 mt-0.5 ml-6">
                            ${canvas.shapes.length} shapes • ${new Date(canvas.updatedAt).toLocaleString('zh-CN', {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'})}
                        </div>
                    </div>
                    <div class="flex items-center gap-1 ml-2">
                        ${!canvas.isCurrent ? `<button onclick="window.switchCanvas(${index})" class="p-1 text-slate-400 hover:text-cyan-400 transition-colors" title="切换到此画布">
                            <i data-lucide="arrow-right" class="w-3 h-3"></i>
                        </button>` : ''}
                        <button onclick="window.renameCanvas(${index})" class="p-1 text-slate-400 hover:text-yellow-400 transition-colors" title="重命名">
                            <i data-lucide="edit" class="w-3 h-3"></i>
                        </button>
                        <button onclick="window.duplicateCanvas(${index})" class="p-1 text-slate-400 hover:text-green-400 transition-colors" title="复制">
                            <i data-lucide="copy" class="w-3 h-3"></i>
                        </button>
                        <button onclick="window.deleteCanvas(${index})" class="p-1 text-slate-400 hover:text-red-400 transition-colors" title="删除">
                            <i data-lucide="trash-2" class="w-3 h-3"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        // 重新初始化 lucide 图标
        if (window.lucide) {
            lucide.createIcons();
        }
    }
    
    // 切换画布
    window.switchCanvas = function(index) {
        if (canvasManager.switchToCanvas(index)) {
            const canvas = canvasManager.getCurrentCanvas();
            
            // 加载该画布的 shapes
            if (window.loadShapes) {
                window.loadShapes(canvas.shapes);
            }
            
            showToast(`✅ 已切换到: ${canvas.name}`);
            hideCanvasMenu();
        }
    };
    
    // 重命名画布
    window.renameCanvas = function(index) {
        const canvas = canvasManager.canvases[index];
        const newName = prompt('请输入新名称:', canvas.name);
        if (newName && newName !== canvas.name) {
            canvasManager.renameCanvas(index, newName);
            showToast(`✅ 已重命名为: ${newName}`);
        }
    };
    
    // 复制画布
    window.duplicateCanvas = function(index) {
        const newCanvas = canvasManager.duplicateCanvas(index);
        if (newCanvas) {
            // 加载复制的画布
            if (window.loadShapes) {
                window.loadShapes(newCanvas.shapes);
            }
            showToast(`✅ 已复制画布: ${newCanvas.name}`);
            hideCanvasMenu();
        }
    };
    
    // 删除画布
    window.deleteCanvas = function(index) {
        const canvas = canvasManager.canvases[index];
        if (confirm(`确定要删除画布"${canvas.name}"吗？`)) {
            if (canvasManager.deleteCanvas(index)) {
                // 如果删除的是当前画布，加载新的当前画布
                if (index === canvasManager.currentCanvasIndex || canvasManager.currentCanvasIndex < 0) {
                    const currentCanvas = canvasManager.getCurrentCanvas();
                    if (currentCanvas && window.loadShapes) {
                        window.loadShapes(currentCanvas.shapes);
                    }
                }
                showToast(`✅ 已删除: ${canvas.name}`);
            }
        }
    };
    
    // 导出所有画布
    function exportAllCanvases() {
        const data = canvasManager.exportAllCanvases();
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `feynman-diagrams-${new Date().toISOString().slice(0,10)}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        showToast('✅ 已导出所有画布');
        hideCanvasMenu();
    }
    
    // 导入画布
    function importCanvases() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'application/json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);
                    canvasManager.importCanvases(data);
                    
                    // 加载第一个画布
                    const currentCanvas = canvasManager.getCurrentCanvas();
                    if (currentCanvas && window.loadShapes) {
                        window.loadShapes(currentCanvas.shapes);
                    }
                    
                    showToast(`✅ 已导入 ${data.canvases.length} 个画布`);
                    hideCanvasMenu();
                } catch (error) {
                    alert('导入失败：文件格式错误');
                    console.error(error);
                }
            };
            reader.readAsText(file);
        };
        
        input.click();
    }
    
    // Toast 提示
    function showToast(message) {
        if (window.showToast) {
            window.showToast(message);
        } else {
            console.log(message);
        }
    }
    
    // 暴露给全局
    window.canvasManagerInstance = null;
    window.initCanvasManagerUI = initCanvasManagerUI;
    window.getCanvasManager = () => canvasManager;
    
    // 页面加载完成后自动初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCanvasManagerUI);
    } else {
        initCanvasManagerUI();
    }
})();
