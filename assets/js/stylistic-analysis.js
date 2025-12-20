// Formality Score Chart
const formalityCtx = document.getElementById('formalityScoreChart');
const formalityXs = [14.5, 17.3, 20.0, 22.8, 25.5, 28.3, 31.0, 33.8, 36.5, 39.3, 42.0, 44.8, 47.5, 50.3, 53.0, 55.8, 58.6, 61.3, 64.1, 66.8, 69.6, 72.3, 75.1, 77.8, 80.6, 83.3];
const formalityNy = [0, 0, 0, 0, 0, 0, 0, 0, 0.0017, 0.0252, 0.0621, 0.1175, 0.1024, 0.0722, 0.0235, 0.0118, 0.0034, 0, 0, 0, 0, 0, 0, 0, 0];
const formalityOx = [0.0072, 0.0084, 0.0096, 0.0132, 0.0180, 0.0265, 0.0277, 0.0445, 0.0373, 0.0409, 0.0373, 0.0421, 0.0253, 0.0277, 0.0217, 0.0084, 0.0120, 0.0060, 0.0012, 0.0012, 0.0012, 0.0012, 0, 0, 0.0012];

new Chart(formalityCtx, {
    type: 'bar',
    data: {
        labels: formalityXs,
        datasets: [
            {
                label: 'NYCC',
                data: formalityNy.map(y => y * 2.75),
                backgroundColor: '#D10000' // New Yorker color updated
            },
            {
                label: 'OHIC',
                data: formalityOx.map(y => y * 2.75),
                backgroundColor: '#002147' // Oxford color updated
            }
        ]
    },
    options: {
        scales: {
            x: {
                title: { display: true, text: 'Formality Score', font: { size: 16 }}
            },
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Density (%)', font: { size: 16 }}
            }
        },
        plugins: {
            title: { 
                display: true, 
                text: 'Formality Score Distributions',
                font: { size: 18 } // Increased title size
            },
            tooltip: {
                callbacks: {
                    label: context => `F-score: ${context.label}, Density: ${(context.raw).toFixed(2)}%`
                }
            }
        }
    }
});

// Part-of-Speech Distribution Chart
const posCtx = document.getElementById('posDistributionChart');
const posLabels = ['Noun', 'Adjective', 'Preposition', 'Article', 'Pronoun', 'Verb', 'Adverb', 'Interjection', 'Conjunction', 'Unknown'];
const posNy = [0.184, 0.052, 0.067, 0.081, 0.131, 0.211, 0.072, 0.011, 0.016, 0.176];
const posOx = [0.133, 0.041, 0.056, 0.067, 0.138, 0.202, 0.053, 0.048, 0.013, 0.250];

new Chart(posCtx, {
    type: 'bar',
    data: {
        labels: posLabels,
        datasets: [
            {
                label: 'NYCC',
                data: posNy,
                backgroundColor: '#D10000' // New Yorker color updated
            },
            {
                label: 'OHIC',
                data: posOx,
                backgroundColor: '#002147' // Oxford color updated
            }
        ]
    },
    options: {
        scales: {
            x: {
                title: { display: true, text: 'Part of Speech', font: { size: 16 }}
            },
            y: {
                beginAtZero: true,
                title: { display: true, text: 'Density (%)', font: { size: 16 }}
            }
        },
        plugins: {
            title: { 
                display: true, 
                text: 'Part-of-Speech Distributions',
                font: { size: 18 } // Increased title size
            },
            tooltip: {
                callbacks: {
                    label: context => `${context.label}: ${(context.raw * 100).toFixed(1)}%`
                }
            }
        }
    }
});