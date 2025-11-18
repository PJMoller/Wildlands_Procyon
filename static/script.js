document.addEventListener('DOMContentLoaded', () => {

    const chartButtons = document.querySelectorAll('.chart-options button');
    const startDateInput = document.getElementById('start-date');

    // Default = today
    startDateInput.value = new Date().toISOString().split('T')[0];

    // Load initial charts
    loadChart('week', startDateInput.value);
    loadTodayWidgets();
    loadDaySummary(startDateInput.value);

    // Range buttons
    chartButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            chartButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadChart(btn.dataset.range, startDateInput.value);
        });
    });

    // When date changes → update attendance + pie chart
    startDateInput.addEventListener('change', () => {
        const activeBtn = document.querySelector('.chart-options button.active');
        loadChart(activeBtn.dataset.range, startDateInput.value);
        loadDaySummary(startDateInput.value);
    });
});


// ---------------------------------------------------------
//  Expected Attendance CHART
// ---------------------------------------------------------
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
            autosize: true,
            responsive: true
        };

        Plotly.newPlot(chartEl, [trace], layout, { responsive: true, displayModeBar: false });

    } catch (error) {
        console.error('Error loading chart:', error);
    }
}


// ---------------------------------------------------------
//  Today + Tomorrow widgets
// ---------------------------------------------------------
async function loadTodayWidgets() {
    const res = await fetch("/api/today");
    const data = await res.json();

    document.querySelector(".current-day .content-box p:first-child").textContent = data.today.date;
    document.querySelector(".current-day .visitor-count").textContent =
        (data.today.visitors?.toLocaleString() || "No data") + " Visitors";

    document.querySelector(".tomorrow .content-box p:first-child").textContent = data.tomorrow.date;
    document.querySelector(".tomorrow .visitor-count").textContent =
        (data.tomorrow.visitors?.toLocaleString() || "No data") + " Visitors";
}


// ---------------------------------------------------------
//  PIE CHART for Selected Day (Top 5 + Overig)
// ---------------------------------------------------------
async function loadDaySummary(date) {
    try {
        const res = await fetch("/api/visitors?range=day&date=" + date);
        const data = await res.json();

        // Validate
        if (!data || !data.visitors || data.visitors.length === 0) {
            console.warn("No data for selected day");
            return;
        }

        // The backend returns only total_visitors for "day" range.
        // We need a separate endpoint or extra data embedded.
        // Assuming your backend now returns ticket breakdown too:

        const summaryRes = await fetch("/api/day-tickets?date=" + date);
        const summary = await summaryRes.json();

        if (!summary || !summary.tickets) return;

        let entries = Object.entries(summary.tickets);

        // Sort by value descending
        entries.sort((a, b) => b[1] - a[1]);

        // Top 5 + rest combined
        const top5 = entries.slice(0, 5);
        const rest = entries.slice(5);
        const restTotal = rest.reduce((sum, x) => sum + x[1], 0);

        if (restTotal > 0) top5.push(["Overig", restTotal]);

        const labels = top5.map(x => x[0]);
        const values = top5.map(x => x[1]);

        // Forecast text
        document.querySelector(".forecast .visitor-count").textContent =
            summary.total_visitors.toLocaleString() + " visitors";

        // PIE chart
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
        console.error("Error loading summary / pie chart:", error);
    }
}
