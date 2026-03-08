document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const input = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const searchBtn = document.getElementById('search-btn');
    const searchSpinner = document.getElementById('search-spinner');
    const searchSpan = searchBtn.querySelector('span');

    const metaBar = document.getElementById('search-meta');
    const metaTime = document.getElementById('meta-time');
    const metaCluster = document.getElementById('meta-cluster-id');
    const cacheBadge = document.getElementById('cache-badge');
    const viewUiBtn = document.getElementById('view-ui-btn');
    const viewJsonBtn = document.getElementById('view-json-btn');
    const jsonContainer = document.getElementById('json-container');
    const jsonOutput = document.getElementById('json-output');

    const statHitRate = document.getElementById('stat-hit-rate');
    const statEntries = document.getElementById('stat-entries');
    const statHits = document.getElementById('stat-hits');
    const statMisses = document.getElementById('stat-misses');
    const statTime = document.getElementById('stat-time');
    const statGrid = document.querySelector('.stat-grid');
    const statsTimeMetric = document.getElementById('stats-time-metric');
    const statsUiBtn = document.getElementById('stats-ui-btn');
    const statsJsonBtn = document.getElementById('stats-json-btn');
    const statsJsonContainer = document.getElementById('stats-json-container');
    const statsJsonOutput = document.getElementById('stats-json-output');

    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const flushBtn = document.getElementById('flush-btn');

    const API_URL = 'http://127.0.0.1:8000';

    fetchStats();

    thresholdSlider.addEventListener('input', (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    viewUiBtn.addEventListener('click', () => {
        viewUiBtn.classList.add('active');
        viewJsonBtn.classList.remove('active');
        resultsContainer.classList.remove('hidden');
        jsonContainer.classList.add('hidden');
    });

    viewJsonBtn.addEventListener('click', () => {
        viewJsonBtn.classList.add('active');
        viewUiBtn.classList.remove('active');
        jsonContainer.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    });

    statsUiBtn.addEventListener('click', () => {
        statsUiBtn.classList.add('active');
        statsJsonBtn.classList.remove('active');
        statGrid.classList.remove('hidden');
        statsTimeMetric.classList.remove('hidden');
        statsJsonContainer.classList.add('hidden');
    });

    statsJsonBtn.addEventListener('click', () => {
        statsJsonBtn.classList.add('active');
        statsUiBtn.classList.remove('active');
        statGrid.classList.add('hidden');
        statsTimeMetric.classList.add('hidden');
        statsJsonContainer.classList.remove('hidden');
    });

    async function fetchStats() {
        try {
            const res = await fetch(`${API_URL}/cache/stats`);
            const data = await res.json();

            statHitRate.textContent = `${(data.hit_rate * 100).toFixed(1)}%`;
            statEntries.textContent = data.total_entries;
            statHits.textContent = data.hit_count;
            statMisses.textContent = data.miss_count;
            statTime.textContent = `${data.avg_lookup_ms} ms`;

            statsJsonOutput.textContent = JSON.stringify(data, null, 2);

            if (data.threshold !== parseFloat(thresholdSlider.value)) {
                thresholdSlider.value = data.threshold;
                thresholdVal.textContent = data.threshold.toFixed(2);
            }
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        searchSpan.classList.add('hidden');
        searchSpinner.classList.remove('hidden');
        metaBar.classList.add('hidden');
        cacheBadge.classList.add('hidden');

        let startTime = performance.now();

        try {
            const threshold = parseFloat(thresholdSlider.value);

            const res = await fetch(`${API_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, threshold })
            });

            const data = await res.json();
            renderResults(data, performance.now() - startTime);
            fetchStats();

        } catch (error) {
            console.error('Search failed:', error);
            resultsContainer.innerHTML = `
                <div class="empty-state" style="color:var(--accent-danger)">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 48px; margin-bottom: 20px;"></i>
                    <h2>Connection Error</h2>
                    <p>Failed to connect to the neural engine. Make sure the FastAPI server is running.</p>
                </div>
            `;
        } finally {
            searchSpan.classList.remove('hidden');
            searchSpinner.classList.add('hidden');
        }
    });

    flushBtn.addEventListener('click', async () => {
        const originalText = flushBtn.innerHTML;
        flushBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Flushing...';
        flushBtn.disabled = true;

        try {
            await fetch(`${API_URL}/cache`, { method: 'DELETE' });
            await fetchStats();
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="animated-icon"><i class="fa-solid fa-satellite-dish"></i></div>
                    <h2>Cache Flushed</h2>
                    <p>All embeddings removed from cache. System ready.</p>
                </div>
            `;
            jsonOutput.textContent = '';
            metaBar.classList.add('hidden');
        } catch (e) {
            console.error("Flush failed", e);
        } finally {
            flushBtn.innerHTML = originalText;
            flushBtn.disabled = false;
        }
    });

    function renderResults(data, durationMs) {
        metaBar.classList.remove('hidden');
        metaTime.textContent = `${durationMs.toFixed(0)} ms`;
        metaCluster.textContent = data.dominant_cluster;

        if (data.cache_hit) {
            cacheBadge.classList.remove('hidden');
        } else {
            cacheBadge.classList.add('hidden');
        }

        jsonOutput.textContent = JSON.stringify(data, null, 2);

        const text = data.result;

        if (text === "No results found." || text.startsWith("Vector store not available")) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-ghost" style="font-size: 48px; margin-bottom: 20px;"></i>
                    <h2>No Match Found</h2>
                    <p>${text}</p>
                </div>
            `;
            return;
        }

        const lines = text.split('\n');
        let html = '';
        let current = null;

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i];
            if (!line.trim()) continue;

            const match = line.match(/^\d+\.\s+\[(.*?)\]\s+\(similarity:\s+(.*?)\)/);

            if (match) {
                if (current) html += renderCard(current.category, current.sim, current.text, current.index);
                current = { category: match[1], sim: parseFloat(match[2]), text: '', index: i };
            } else if (current) {
                current.text += line.trim() + ' ';
            }
        }
        if (current) html += renderCard(current.category, current.sim, current.text, current.index);

        resultsContainer.innerHTML = html;

        setTimeout(() => {
            document.querySelectorAll('.sim-bar-fill').forEach(fill => {
                fill.style.width = fill.getAttribute('data-width');
            });
        }, 50);
    }

    function renderCard(category, similarity, text, index) {
        let hash = 0;
        for (let i = 0; i < category.length; i++) {
            hash = category.charCodeAt(i) + ((hash << 5) - hash);
        }
        const hue = Math.abs(hash) % 360;
        const catColor = `hsl(${hue}, 70%, 65%)`;
        const simPercent = Math.round(similarity * 100);
        const delay = index * 0.1;

        return `
        <div class="result-card" style="animation-delay: ${delay}s">
            <div class="card-header">
                <span class="card-category" style="color: ${catColor}; border: 1px solid ${catColor}40">${category}</span>
                <div class="similarity-meter">
                    <span>${simPercent}% Match</span>
                    <div class="sim-bar-bg">
                        <div class="sim-bar-fill" data-width="${simPercent}%" style="width: 0%; background: ${catColor}"></div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                ${text}
            </div>
        </div>
        `;
    }
});
