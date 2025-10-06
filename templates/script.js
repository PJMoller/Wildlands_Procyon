// Simple interactivity for the dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Chart options buttons
    const chartButtons = document.querySelectorAll('.chart-options button');
    chartButtons.forEach(button => {
        button.addEventListener('click', function() {
            chartButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Calendar day selection
    const calendarDays = document.querySelectorAll('.calendar-day');
    calendarDays.forEach(day => {
        day.addEventListener('click', function() {
            calendarDays.forEach(d => d.classList.remove('selected'));
            this.classList.add('selected');
            
            // Update forecast based on selected day
            updateForecast(this.textContent);
        });
    });
    
    // Calendar navigation
    const calendarNavButtons = document.querySelectorAll('.calendar-nav button');
    const calendarMonth = document.querySelector('.calendar-month');
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December'];
    
    let currentMonthIndex = 0; // January
    
    calendarNavButtons[0].addEventListener('click', function() {
        // Previous month
        currentMonthIndex = (currentMonthIndex - 1 + 12) % 12;
        calendarMonth.textContent = months[currentMonthIndex];
        updateCalendarDays(currentMonthIndex);
    });
    
    calendarNavButtons[1].addEventListener('click', function() {
        // Next month
        currentMonthIndex = (currentMonthIndex + 1) % 12;
        calendarMonth.textContent = months[currentMonthIndex];
        updateCalendarDays(currentMonthIndex);
    });
    
    // Confirm day button
    const confirmButton = document.querySelector('.confirm-button');
    confirmButton.addEventListener('click', function() {
        const selectedDay = document.querySelector('.calendar-day.selected');
        if (selectedDay) {
            alert(`Day ${selectedDay.textContent} ${calendarMonth.textContent} confirmed!`);
        } else {
            alert('Please select a day first!');
        }
    });
    
    // Manual link
    const manualLink = document.querySelector('.manual-button');
    manualLink.addEventListener('click', function(e) {
        e.preventDefault();
        alert('Redirecting to manual slider page...');
        // In a real application, this would navigate to the actual page
    });
});

function updateForecast(day) {
    // Simple forecast calculation based on day
    const baseVisitors = 4570;
    const variation = (parseInt(day) % 10) * 50; // Simple variation based on day
    const forecastVisitors = baseVisitors + variation;
    
    const forecastElement = document.querySelector('.forecast .visitor-count');
    forecastElement.textContent = `${forecastVisitors.toLocaleString()} visitors`;
}

function updateCalendarDays(monthIndex) {
    // This is a simplified version - in a real app you would generate
    // the correct number of days for each month
    const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    const calendarGrid = document.querySelector('.calendar-grid');
    
    // Clear existing days
    calendarGrid.innerHTML = '';
    
    // Add days for the month
    for (let i = 1; i <= daysInMonth[monthIndex]; i++) {
        const dayElement = document.createElement('div');
        dayElement.className = 'calendar-day';
        dayElement.textContent = i;
        calendarGrid.appendChild(dayElement);
    }
    
    // Reattach event listeners to new days
    const newCalendarDays = document.querySelectorAll('.calendar-day');
    newCalendarDays.forEach(day => {
        day.addEventListener('click', function() {
            newCalendarDays.forEach(d => d.classList.remove('selected'));
            this.classList.add('selected');
            updateForecast(this.textContent);
        });
    });
}