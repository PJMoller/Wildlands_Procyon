document.addEventListener('DOMContentLoaded', () => {
    const chartButtons = document.querySelectorAll('.chart-options button');
    const startDateInput = document.getElementById('start-date');
    const ticketDateInput = document.getElementById('ticket-date');

    // Default date = today
    const today = new Date().toISOString().split('T')[0];
    startDateInput.value = today;
    ticketDateInput.value = today;

    // Load default data
    loadChart('week', startDateInput.value);
    loadTodayWidgets();
    loadTicketPie(ticketDateInput.value);

    // Handle attendance range buttons
    chartButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            chartButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadChart(btn.dataset.range, startDateInput.value);
        });
    });

    // Handle attendance date change
    startDateInput.addEventListener('change', () => {
        const activeBtn = document.querySelector('.chart-options button.active');
        loadChart(activeBtn.dataset.range, startDateInput.value);
    });

    // Handle ticket pie date change
    ticketDateInput.addEventListener('change', () => {
        loadTicketPie(ticketDateInput.value);
    });
});


async function loadChart(range = 'week', date) {
    try {
        const res = await fetch(`/api/visitors?range=${range}&date=${date}`);
        const data = await res.json();
        const chartEl = document.getElementById('attendanceChart');

        if (!data.visitors || data.visitors.length === 0) {
            chartEl.innerHTML = '<p>No data available for this range.</p>';
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
            font: { color: '#000000', family: 'Arial, sans-serif' },
            margin: { l: 40, r: 20, t: 40, b: 40 },
            xaxis: { tickangle: -45, showgrid: false, zeroline: false, title: { text: 'Date' } },
            yaxis: { showgrid: true, gridcolor: '#444', title: { text: 'Visitors' } },
            bargap: 0.25,
            autosize: true,
            responsive: true
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.newPlot(chartEl, [trace], layout, config);
    } catch (error) {
        console.error('Error loading chart:', error);
    }
}


async function loadTodayWidgets() {
    const res = await fetch("/api/today");
    const data = await res.json();

    // Current day
    document.querySelector(".current-day .content-box p:first-child").textContent = data.today.date;
    document.querySelector(".current-day .visitor-count").textContent =
        (data.today.visitors?.toLocaleString() || "No data") + " Visitors";

    // Tomorrow
    document.querySelector(".tomorrow .content-box p:first-child").textContent = data.tomorrow.date;
    document.querySelector(".tomorrow .visitor-count").textContent =
        (data.tomorrow.visitors?.toLocaleString() || "No data") + " Visitors";
}


async function loadTicketPie(date) {
    try {
        const res = await fetch(`/api/tickets?date=${date}`);
        const data = await res.json();
        const chartEl = document.getElementById('ticketPieChart');

        if (!data.values || data.values.length === 0) {
            chartEl.innerHTML = '<p>No ticket data available for this date.</p>';
            return;
        }

        const trace = {
            labels: data.labels,
            values: data.values,
            type: 'pie',
            textinfo: 'label+percent',
            hoverinfo: 'label+value+percent',
            hole: 0.3
        };

        const layout = {
            paper_bgcolor: '#2b2b2b',
            plot_bgcolor: '#2b2b2b',
            font: { color: '#000000', family: 'Arial, sans-serif' },
            margin: { t: 40, b: 20, l: 20, r: 20 }
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.newPlot(chartEl, [trace], layout, config);
    } catch (err) {
        console.error('Error loading ticket pie:', err);
    }
}
