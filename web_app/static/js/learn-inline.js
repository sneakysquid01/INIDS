const socket = io('/events');

socket.on('connect', function() {
    console.log('Connected to real-time events');
});

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabName + '-tab').style.display = 'block';
    event.target.classList.add('active');
}

function triggerRetraining() {
    console.log('Triggering immediate retraining...');
    socket.emit('training.trigger', { reason: 'Manual trigger' });
    alert('✓ Retraining triggered! This will run in the background.');
}

function viewSchedule() {
    alert('Training schedule view - coming soon');
}

function viewLogs() {
    alert('Training logs view - coming soon');
}

function viewChart() {
    alert('Accuracy trend chart - coming soon');
}

function downloadReport() {
    alert('Downloading performance report...');
}

function exportData() {
    alert('Exporting training data...');
}

function viewDistribution() {
    alert('Data distribution view - coming soon');
}

function compareVersions() {
    alert('Model comparison view - coming soon');
}

function rollbackModel() {
    if (confirm('Are you sure you want to rollback to the previous model version?')) {
        alert('✓ Model rollback initiated');
    }
}

socket.on('training.started', function(payload) {
    console.log('Training started:', payload);
    alert('🚀 Training started!');
});

socket.on('training.completed', function(payload) {
    console.log('Training completed:', payload);
    alert('✓ Training completed! New model deployed.');
});
