document.querySelectorAll('.quiz-option').forEach(option => {
    option.addEventListener('click', () => {
        const currentStep = option.closest('.quiz-step'); // Find the current quiz step
        const feedbackElement = currentStep.querySelector('.feedback');
        const feedbackText = option.getAttribute('data-feedback'); // Get the tailored feedback

        // Display feedback
        if (feedbackElement) {
            feedbackElement.textContent = feedbackText;
            feedbackElement.classList.remove('hidden');
            feedbackElement.classList.add('visible');
        }

        // Disable all options in the current step
        currentStep.querySelectorAll('.quiz-option').forEach(option => {
            option.classList.add('disabled'); // Add a disabled class
            option.style.pointerEvents = 'none'; // Prevent further clicks
        });
    });
});