document.addEventListener('DOMContentLoaded', () => {

    const chartButtons = document.querySelectorAll('.chart-options button');
    const startDateInput = document.getElementById('start-date');

    if (!startDateInput) return;

    startDateInput.value = new Date().toISOString().split('T')[0];

    loadChart('week', startDateInput.value);
    loadTodayWidgets();
    loadDaySummary(startDateInput.value);
    loadUploadStatus();
    loadEvents(); // ✅ ADDED

    chartButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            chartButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadChart(btn.dataset.range, startDateInput.value);
        });
    });

    startDateInput.addEventListener('change', () => {
        const activeBtn = document.querySelector('.chart-options button.active');
        loadChart(activeBtn.dataset.range, startDateInput.value);
        loadDaySummary(startDateInput.value);
    });
});


// ---------------- EVENTS (ADDED) ----------------
async function loadEvents() {
    const container = document.getElementById("events-container");
    if (!container) return;

    const res = await fetch("/api/events");
    const data = await res.json();

    container.innerHTML = "";

    if (!data.events.length) {
        container.innerHTML = "<p>No upcoming events</p>";
        return;
    }

    data.events.forEach(e => {
        container.innerHTML += `
            <div class="event ${e.status}">
                <span class="event-status">${e.status.toUpperCase()}!</span>
                <div class="event-content">
                    <p class="event-title">${e.event_name}, ${e.date}</p>
                    <p class="event-description">
                        Impact: ${e.impact > 0 ? "+" : ""}${e.impact} visitors
                    </p>
                </div>
            </div>
        `;
    });
}


// ---------------- WEATHER ----------------
async function loadTodayWidgets() {
    const res = await fetch("/api/today");
    const data = await res.json();

    const todayBox = document.querySelector(".current-day .content-box");
    const tomorrowBox = document.querySelector(".tomorrow .content-box");

    todayBox.querySelector(".visitor-count").textContent =
        (data.today.visitors?.toLocaleString() || "0") + " Visitors";

    tomorrowBox.querySelector(".visitor-count").textContent =
        (data.tomorrow.visitors?.toLocaleString() || "0") + " Visitors";

    const todayWeather = `🌡 ${data.today.temperature ?? "?"}°C · 🌧 ${data.today.rain ?? "?"} mm`;
    const tomorrowWeather = `🌡 ${data.tomorrow.temperature ?? "?"}°C · 🌧 ${data.tomorrow.rain ?? "?"} mm`;

    addWeatherText(todayBox, todayWeather);
    addWeatherText(tomorrowBox, tomorrowWeather);
}

function addWeatherText(container, text) {
    let weatherEl = container.querySelector(".weather-info");

    if (!weatherEl) {
        weatherEl = document.createElement("p");
        weatherEl.className = "weather-info";
        container.appendChild(weatherEl);
    }

    weatherEl.textContent = text;
}


// ---------------- BAR CHART ----------------
async function loadChart(range = 'week', date) {
    try {
        const res = await fetch(`/api/visitors?range=${range}&date=${date}`);
        const data = await res.json();
        const chartEl = document.getElementById('attendanceChart');

        if (!data.visitors || data.visitors.length === 0) {
            chartEl.innerHTML = '<p>No data available.</p>';
            return;
        }

        const trace = {
            x: data.dates,
            y: data.visitors,
            type: 'bar',
            marker: {
                color: '#2196F3',
                line: { color: '#1976D2', width: 1.5 }
            },
            hovertemplate: '%{x}<br><b>%{y}</b> visitors<extra></extra>'
        };

        const layout = {
            plot_bgcolor: '#2b2b2b',
            paper_bgcolor: '#2b2b2b',
            font: { color: '#ffffff', family: 'Arial, sans-serif' },
            margin: { l: 40, r: 20, t: 40, b: 40 },
            xaxis: { tickangle: -45, showgrid: false, zeroline: false },
            yaxis: { showgrid: true, gridcolor: '#444' },
            bargap: 0.25,
            autosize: true
        };

        Plotly.newPlot(chartEl, [trace], layout, { responsive: true, displayModeBar: false });

    } catch (error) {
        console.error(error);
    }
}


// ---------------- PIE CHART ----------------
async function loadDaySummary(date) {
    try {
        const summaryRes = await fetch("/api/day-tickets?date=" + date);
        const summary = await summaryRes.json();

        if (!summary || !summary.tickets) return;

        let entries = Object.entries(summary.tickets);
        entries.sort((a, b) => b[1] - a[1]);

        const top5 = entries.slice(0, 5);
        const rest = entries.slice(5);
        const restTotal = rest.reduce((sum, x) => sum + x[1], 0);

        if (restTotal > 0) top5.push(["Overig", restTotal]);

        const labels = top5.map(x => x[0]);
        const values = top5.map(x => x[1]);

        document.querySelector(".forecast .visitor-count").textContent =
            summary.total_visitors.toLocaleString() + " visitors";

        const trace = {
            labels: labels,
            values: values,
            type: "pie",
            textposition: "inside",
            textinfo: "label+percent",
            insidetextorientation: "radial",
            hovertemplate: "%{label}<br>%{value} tickets<extra></extra>",
            textfont: { size: 10 },
            automargin: true
        };

        const layout = {
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: false
        };

        Plotly.newPlot("pieChart", [trace], layout, { displayModeBar: false });

    } catch (error) {
        console.error(error);
    }
}


// ---------------- UPLOAD STATUS ----------------
async function loadUploadStatus() {
    const statusEl = document.getElementById("upload-status");
    if (!statusEl) return;

    const res = await fetch("/api/upload-status");
    const data = await res.json();

    if (!data.files.length) {
        statusEl.innerHTML = `<p>${translations.no_files}</p>`;
    } else {
        statusEl.innerHTML = `
            <p><strong>${translations.uploaded_files}</strong></p>
            <ul>${data.files.map(f => `<li>${f}</li>`).join("")}</ul>
        `;
    }
}


// ---------------- FILE INPUT UI ----------------
const fileInput = document.querySelector('input[type="file"]');
const fileName = document.querySelector('.file-name');

if (fileInput) {
    fileInput.addEventListener('change', () => {
        fileName.textContent = fileInput.files.length
            ? fileInput.files[0].name
            : 'No file chosen';
    });
}
