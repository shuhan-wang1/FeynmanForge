/**
 * FeynmanForge Canvas Manager
 * 多画布管理系统 - 支持创建、切换、保存多个 Feynman 图画布
 */

class CanvasManager {
    constructor() {
        this.canvases = [];
        this.currentCanvasIndex = -1;
        this.nextCanvasId = 1;
        
        // 创建第一个默认画布
        this.createNewCanvas('Untitled Diagram 1');
    }
    
    /**
     * 创建新画布
     */
    createNewCanvas(name = null, shapes = []) {
        const canvas = {
            id: this.nextCanvasId++,
            name: name || `Untitled Diagram ${this.nextCanvasId}`,
            shapes: shapes,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        this.canvases.push(canvas);
        this.currentCanvasIndex = this.canvases.length - 1;
        
        console.log(`📄 创建新画布: ${canvas.name} (ID: ${canvas.id})`);
        this.saveToLocalStorage();
        this.notifyUpdate();
        
        return canvas;
    }
    
    /**
     * 切换到指定画布
     */
    switchToCanvas(index) {
        if (index < 0 || index >= this.canvases.length) {
            console.error('❌ 无效的画布索引:', index);
            return false;
        }
        
        this.currentCanvasIndex = index;
        console.log(`🔄 切换到画布: ${this.getCurrentCanvas().name}`);
        this.notifyUpdate();
        return true;
    }
    
    /**
     * 通过 ID 切换画布
     */
    switchToCanvasById(id) {
        const index = this.canvases.findIndex(c => c.id === id);
        return this.switchToCanvas(index);
    }
    
    /**
     * 获取当前画布
     */
    getCurrentCanvas() {
        if (this.currentCanvasIndex < 0 || this.currentCanvasIndex >= this.canvases.length) {
            return null;
        }
        return this.canvases[this.currentCanvasIndex];
    }
    
    /**
     * 更新当前画布的 shapes
     */
    updateCurrentCanvas(shapes) {
        const canvas = this.getCurrentCanvas();
        if (!canvas) return;
        
        canvas.shapes = shapes;
        canvas.updatedAt = new Date().toISOString();
        this.saveToLocalStorage();
    }
    
    /**
     * 重命名画布
     */
    renameCanvas(index, newName) {
        if (index < 0 || index >= this.canvases.length) return false;
        
        this.canvases[index].name = newName;
        this.canvases[index].updatedAt = new Date().toISOString();
        this.saveToLocalStorage();
        this.notifyUpdate();
        return true;
    }
    
    /**
     * 删除画布
     */
    deleteCanvas(index) {
        if (this.canvases.length <= 1) {
            alert('至少需要保留一个画布！');
            return false;
        }
        
        if (index < 0 || index >= this.canvases.length) return false;
        
        const deletedCanvas = this.canvases[index];
        this.canvases.splice(index, 1);
        
        // 调整当前索引
        if (this.currentCanvasIndex >= this.canvases.length) {
            this.currentCanvasIndex = this.canvases.length - 1;
        } else if (this.currentCanvasIndex > index) {
            this.currentCanvasIndex--;
        }
        
        console.log(`🗑️ 删除画布: ${deletedCanvas.name}`);
        this.saveToLocalStorage();
        this.notifyUpdate();
        return true;
    }
    
    /**
     * 复制画布
     */
    duplicateCanvas(index) {
        if (index < 0 || index >= this.canvases.length) return null;
        
        const original = this.canvases[index];
        const duplicate = {
            id: this.nextCanvasId++,
            name: `${original.name} (Copy)`,
            shapes: JSON.parse(JSON.stringify(original.shapes)), // 深拷贝
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        this.canvases.splice(index + 1, 0, duplicate);
        this.currentCanvasIndex = index + 1;
        
        console.log(`📋 复制画布: ${original.name} → ${duplicate.name}`);
        this.saveToLocalStorage();
        this.notifyUpdate();
        return duplicate;
    }
    
    /**
     * 获取所有画布列表
     */
    getAllCanvases() {
        return this.canvases.map((c, i) => ({
            ...c,
            index: i,
            isCurrent: i === this.currentCanvasIndex
        }));
    }
    
    /**
     * 保存到 LocalStorage
     */
    saveToLocalStorage() {
        try {
            localStorage.setItem('feynmanforge_canvases', JSON.stringify({
                canvases: this.canvases,
                currentCanvasIndex: this.currentCanvasIndex,
                nextCanvasId: this.nextCanvasId
            }));
        } catch (error) {
            console.error('❌ 保存画布失败:', error);
        }
    }
    
    /**
     * 从 LocalStorage 加载
     */
    loadFromLocalStorage() {
        try {
            const saved = localStorage.getItem('feynmanforge_canvases');
            if (!saved) return false;
            
            const data = JSON.parse(saved);
            this.canvases = data.canvases || [];
            this.currentCanvasIndex = data.currentCanvasIndex || 0;
            this.nextCanvasId = data.nextCanvasId || 1;
            
            // 确保至少有一个画布
            if (this.canvases.length === 0) {
                this.createNewCanvas('Untitled Diagram 1');
            }
            
            console.log(`📂 加载了 ${this.canvases.length} 个画布`);
            return true;
        } catch (error) {
            console.error('❌ 加载画布失败:', error);
            return false;
        }
    }
    
    /**
     * 清空所有画布
     */
    clearAll() {
        if (!confirm('确定要清空所有画布吗？此操作无法撤销！')) {
            return false;
        }
        
        this.canvases = [];
        this.currentCanvasIndex = -1;
        this.nextCanvasId = 1;
        this.createNewCanvas('Untitled Diagram 1');
        this.saveToLocalStorage();
        this.notifyUpdate();
        return true;
    }
    
    /**
     * 导出所有画布（用于备份）
     */
    exportAllCanvases() {
        return {
            version: '1.0',
            exportedAt: new Date().toISOString(),
            canvases: this.canvases
        };
    }
    
    /**
     * 导入画布（从备份恢复）
     */
    importCanvases(data) {
        if (!data.canvases || !Array.isArray(data.canvases)) {
            throw new Error('Invalid canvas data format');
        }
        
        this.canvases = data.canvases;
        this.currentCanvasIndex = 0;
        this.nextCanvasId = Math.max(...this.canvases.map(c => c.id)) + 1;
        
        this.saveToLocalStorage();
        this.notifyUpdate();
        console.log(`📥 导入了 ${this.canvases.length} 个画布`);
    }
    
    /**
     * 通知更新（触发 UI 刷新）
     */
    notifyUpdate() {
        // 触发自定义事件
        window.dispatchEvent(new CustomEvent('canvasManagerUpdate', {
            detail: {
                canvases: this.getAllCanvases(),
                currentCanvas: this.getCurrentCanvas()
            }
        }));
    }
}

// 导出全局实例
if (typeof window !== 'undefined') {
    window.CanvasManager = CanvasManager;
}
