document.addEventListener('DOMContentLoaded', () => {
    let score = 0; // Initialize the score
    let questionsAnswered = 0; // Counter to track the number of answered quizzes

    // Add event listeners to all quiz options
    document.querySelectorAll('.quiz-option').forEach(option => {
        option.addEventListener('click', () => {
            const currentStep = option.closest('.quiz-step');
            const feedback = currentStep.querySelector('.feedback');
            const isCorrect = option.getAttribute('data-answer') === 'correct';

            // Update the score if the answer is correct
            if (isCorrect) {
                score++;
            }

            // Display feedback
            feedback.textContent = isCorrect ? 'Correct! Great choice!' : 'Incorrect. But let’s move on!';
            feedback.classList.remove('hidden');
            feedback.classList.add('visible');

            // Disable all options in the current step
            currentStep.querySelectorAll('.quiz-option').forEach(option => {
                option.classList.add('disabled');
                option.style.pointerEvents = 'none';
            });

            // Increment the questionsAnswered counter
            questionsAnswered++;

            // Check if all questions have been answered
            if (questionsAnswered === 10) {
                showResults(score);
            }
        });
    });

    // Function to display the results
    function showResults(score) {
        const resultsSection = document.getElementById('results');
        const resultsMessage = document.getElementById('results-message');

        if (!resultsMessage) {
            return;
        }

        // Generate the message based on the score
        if (score === 10) {
            resultsMessage.innerHTML = "Perfect score! You're <strong>hired</strong>! You have the humor profile that matches New Yorker readers.";

            // Set up the Intersection Observer to trigger confetti when the results section is fully visible
            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        triggerConfetti(); // Trigger the confetti animation
                        observer.disconnect(); // Stop observing after the animation is triggered
                    }
                });
            }, { threshold: 1.0 }); // Ensure the entire section is visible

            observer.observe(resultsSection);
        } else if (score >= 8) {
            resultsMessage.innerHTML = "Great job! You scored " + score + "/10. You have a strong sense of humor, you're nearly there!";
        } else if (score >= 6) {
            resultsMessage.innerHTML = "Great job! You scored " + score + "/10. You're funny, but there's still room for improvement.";
        } else if (score >= 4) {
            resultsMessage.innerHTML = "You scored " + score + "/10. Not bad, but maybe stick to practicing your jokes on Twitter for now!";
        } else {
            resultsMessage.innerHTML = "You scored " + score + "/10. Humor might not be your strong suit, but don't give up!";
        }

        // Show the results section
        resultsSection.classList.remove('hidden');
    }

    // Function to trigger the confetti animation
    function triggerConfetti() {
        const duration = 3 * 1000,
        animationEnd = Date.now() + duration,
        defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };

        function randomInRange(min, max) {
        return Math.random() * (max - min) + min;
        }

        const interval = setInterval(function() {
        const timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
            return clearInterval(interval);
        }

        const particleCount = 50 * (timeLeft / duration);

        // since particles fall down, start a bit higher than random
        confetti(
            Object.assign({}, defaults, {
            particleCount,
            origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 },
            })
        );
        confetti(
            Object.assign({}, defaults, {
            particleCount,
            origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 },
            })
        );
        }, 250);
    }
});