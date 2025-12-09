document.addEventListener("DOMContentLoaded", () => {
  const chartArea = document.getElementById("chart-area");
  const buttons = document.querySelectorAll(".chart-options button");
  const dateInput = document.getElementById("start-date");

function loadChart(period, date) {
  const filename = `charts/chart_${period}_${date}.html`;
  console.log("Loading chart:", filename);

  fetch(filename)
    .then(res => {
      if (!res.ok) throw new Error("File not found");
      return res.text();
    })
    .then(html => {
      console.log("Fetched HTML length:", html.length);

      // Insert the HTML without executing scripts
      chartArea.innerHTML = html;

      // Find all <script> tags in the fetched HTML and execute them
      const scripts = chartArea.querySelectorAll("script");
      scripts.forEach(oldScript => {
        const newScript = document.createElement("script");
        if (oldScript.src) {
          // External script
          newScript.src = oldScript.src;
        } else {
          // Inline script
          newScript.textContent = oldScript.textContent;
        }
        document.body.appendChild(newScript);
        oldScript.remove();
      });
    })
    .catch(err => {
      console.error("Error loading chart:", err);
      chartArea.innerHTML = `<p>No chart found for ${period} starting ${date}.</p>`;
    });
}



  // Load initial chart for 2017-10-01
  const defaultDate = dateInput.value;
  loadChart("week", defaultDate);

  // When clicking week/month/year
  buttons.forEach(button => {
    button.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      loadChart(button.dataset.period, dateInput.value);
    });
  });

  // When changing the date input
  dateInput.addEventListener("change", () => {
    const activeButton = document.querySelector(".chart-options button.active");
    loadChart(activeButton.dataset.period, dateInput.value);
  });
});
  