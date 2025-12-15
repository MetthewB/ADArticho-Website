document.querySelectorAll('.quiz-option').forEach(option => {
    option.addEventListener('click', () => {
        const currentStep = option.closest('.quiz-step'); // Find the current quiz step
        const feedback = currentStep.querySelector('.feedback'); // Find the feedback element
        const isCorrect = option.getAttribute('data-answer') === 'correct'; // Check if the answer is correct

        // Display feedback
        if (feedback) {
            feedback.textContent = isCorrect ? 'Correct! Great choice!' : 'Incorrect. But let’s move on!';
            feedback.classList.remove('hidden');
            feedback.classList.add('visible');
        }

        // Disable all options in the current step
        currentStep.querySelectorAll('.quiz-option').forEach(option => {
            option.classList.add('disabled'); // Add a disabled class
            option.style.pointerEvents = 'none'; // Prevent further clicks
        });
    });
});