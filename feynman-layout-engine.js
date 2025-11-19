/**
 * Feynman Layout Engine
 * 纯数学确定的图布局算法，将拓扑数据转换为几何坐标
 */
class FeynmanLayoutEngine {
    constructor(width, height, padding = 80) {
        this.width = width;
        this.height = height;
        this.padding = padding;
    }

    /**
     * 核心布局函数
     * @param {Object} topology { nodes: [], edges: [] }
     */
    calculateLayout(topology) {
        const { nodes, edges } = topology;

        // 1. 分层 (Layering) - 确定 X 轴 (时间)
        // 初态在 Layer 0，末态在 Layer Max
        const layers = this.assignLayers(nodes, edges);

        // 2. 节点排序 (Ordering) - 确定 Y 轴的相对顺序
        // 这一步是为了减少交叉
        this.minimizeCrossings(layers, edges);

        // 3. 坐标分配 (Coordinate Assignment) - 映射到 Canvas 像素
        const positions = this.assignCoordinates(layers, this.width, this.height);

        return positions;
    }

    /**
     * 步骤 1: 分层算法 (基于最长路径)
     * 确定每个节点的时间切片 (Time Slice)
     */
    assignLayers(nodes, edges) {
        // 初始化所有节点
        const nodeMap = new Map(nodes.map(n => [n.id, { ...n, layer: -1 }]));
        
        // 强制约束：初态在层级 0
        nodes.filter(n => n.type === 'initial').forEach(n => {
            const node = nodeMap.get(n.id);
            if (node) node.layer = 0;
        });

        // 拓扑排序 / 传播层级
        // 简单的迭代法：如果 u -> v，则 layer(v) = max(layer(v), layer(u) + 1)
        let changed = true;
        let maxIter = nodes.length * 2; // 防止环路死循环
        
        while (changed && maxIter-- > 0) {
            changed = false;
            edges.forEach(edge => {
                const u = nodeMap.get(edge.source);
                const v = nodeMap.get(edge.target);
                
                if (u && v && u.layer !== -1) {
                    if (v.layer <= u.layer) {
                        v.layer = u.layer + 1;
                        changed = true;
                    }
                }
            });
        }

        // 处理末态：将所有 final 节点强制推到最大层级
        // 先找出当前计算出的最大层级
        let currentMaxLayer = 0;
        for (let node of nodeMap.values()) {
            if (node.layer > currentMaxLayer) currentMaxLayer = node.layer;
        }
        
        // 如果没有中间层级，至少保证 final 在 initial 之后
        if (currentMaxLayer === 0) currentMaxLayer = 1;

        // 将 final 节点设为 currentMaxLayer (或者 +1 确保在最后)
        // 这里我们统一对齐到最右侧
        nodes.filter(n => n.type === 'final').forEach(n => {
            const node = nodeMap.get(n.id);
            if (node) node.layer = currentMaxLayer;
        });

        // 再次检查：如果有孤立点没分配到层级，默认放在中间
        for (let node of nodeMap.values()) {
            if (node.layer === -1) node.layer = Math.floor(currentMaxLayer / 2);
        }
        
        // 重新计算最大层级（可能因为孤立点处理变了，虽然上面逻辑应该覆盖了）
        const finalMaxLayer = Math.max(...Array.from(nodeMap.values()).map(n => n.layer));

        // 按层级分组返回
        const layers = [];
        for (let i = 0; i <= finalMaxLayer; i++) {
            layers[i] = [];
        }
        
        for (let node of nodeMap.values()) {
            layers[node.layer].push(node);
        }
        
        return layers;
    }

    /**
     * 步骤 2: 减少交叉 (重心法 Barycenter Heuristic)
     * 调整每一层内部节点的顺序，使得连线更顺畅
     */
    minimizeCrossings(layers, edges) {
        // 简单实现：根据上一层连接节点的平均位置来排序当前层
        // 迭代几次以收敛
        for (let iter = 0; iter < 3; iter++) {
            // 正向扫描 (从左到右)
            for (let i = 1; i < layers.length; i++) {
                this.orderLayer(layers[i], layers[i-1], edges, 'target', 'source');
            }
            // 反向扫描 (从右到左)
            for (let i = layers.length - 2; i >= 0; i--) {
                this.orderLayer(layers[i], layers[i+1], edges, 'source', 'target');
            }
        }
    }

    orderLayer(currentLayer, neighborLayer, edges, myKey, neighborKey) {
        const neighborMap = new Map(); // neighborId -> index
        neighborLayer.forEach((n, idx) => neighborMap.set(n.id, idx));

        currentLayer.forEach(node => {
            // 找到所有连接的邻居
            const connectedEdges = edges.filter(e => e[myKey] === node.id);
            let sum = 0;
            let count = 0;
            
            connectedEdges.forEach(e => {
                const neighborId = e[neighborKey];
                if (neighborMap.has(neighborId)) {
                    sum += neighborMap.get(neighborId);
                    count++;
                }
            });

            // 计算重心 (Barycenter)
            // 如果没有连接，保持原位 (使用当前索引作为权重的一部分，或者保持相对位置)
            // 这里简单处理：如果没有连接，设为中间值
            node.orderWeight = count > 0 ? sum / count : neighborLayer.length / 2;
        });

        // 根据重心排序
        currentLayer.sort((a, b) => a.orderWeight - b.orderWeight);
    }

    /**
     * 步骤 3: 映射到像素坐标
     */
    assignCoordinates(layers, w, h) {
        const positions = {};
        const layerCount = layers.length;
        
        // X 轴间距 (Time axis)
        // 留出 padding
        const usableWidth = w - 2 * this.padding;
        
        layers.forEach((layerNodes, layerIndex) => {
            // 计算 X: 均匀分布
            // 如果只有2层(0,1)，则 x = padding, x = w - padding
            // layerIndex / (layerCount - 1) 归一化到 0..1
            const xRatio = layerCount > 1 ? layerIndex / (layerCount - 1) : 0.5;
            const x = this.padding + xRatio * usableWidth;

            // 计算 Y: 在每一层内均匀分布
            const nodeCount = layerNodes.length;
            let usableHeight = h - 2 * this.padding;
            
            // 🔧 Ensure minimum vertical spacing
            const minHeightNeeded = nodeCount * 60; // 60px per node
            if (usableHeight < minHeightNeeded) {
                // If canvas is too short, we can't stretch it, but we can use the full height
                // or just accept they might be close. 
                // Better: use the full height if needed, ignoring padding
                if (h > minHeightNeeded) {
                    usableHeight = h - 20; // Minimal padding
                }
            }

            layerNodes.forEach((node, nodeIndex) => {
                // 0.5 用于居中
                // nodeIndex 从 0 到 nodeCount-1
                // (nodeIndex + 1) / (nodeCount + 1) 将区间分成 nodeCount+1 份，取中间的 nodeCount 个点
                const yRatio = (nodeIndex + 1) / (nodeCount + 1);
                let y = this.padding + (yRatio - 0.5) * usableHeight + usableHeight/2;
                
                // 🔧 防止共线问题 (Collinearity Prevention)
                // 如果连续三层都只有一个节点，它们会形成一条直线，导致 validateDiagram 误判为"点在线上"
                // 我们对单节点层引入显著的波浪偏移 (Wave Offset)
                if (nodeCount === 1) {
                    // 使用正弦波产生更自然的偏移，避免简单的 zigzag 导致的偶数层共线
                    // 周期为非整数，避免与层级步长共振。幅度 60px 确保超过 SNAP_DIST
                    const wave = Math.sin(layerIndex * 2.1); 
                    const offset = wave * 60;
                    y += offset;
                } else {
                    // 即使有多个节点，也添加微小的确定性扰动，防止完美对齐
                    // 使用确定性哈希 (基于 ID)
                    let hash = 0;
                    for (let i = 0; i < node.id.length; i++) {
                        hash = ((hash << 5) - hash) + node.id.charCodeAt(i);
                        hash |= 0;
                    }
                    const pseudoRandom = (hash % 21) - 10; // -10 to +10
                    y += pseudoRandom;
                }

                positions[node.id] = { x, y };
            });
        });

        return positions;
    }
}

// 导出供集成使用
window.FeynmanLayoutEngine = FeynmanLayoutEngine;
