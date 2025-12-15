document.addEventListener('DOMContentLoaded', () => {
    let score = 0; // Initialize the score
    let questionsAnswered = 0; // Counter to track the number of answered quizzes

    // Add event listeners to all quiz options
    document.querySelectorAll('.quiz-option').forEach(option => {
        option.addEventListener('click', () => {
            const currentStep = option.closest('.quiz-step');
            const feedbackElement = currentStep.querySelector('.feedback');
            const feedbackText = option.getAttribute('data-feedback');
            const isCorrect = option.getAttribute('data-answer') === 'correct';

            // Update the score if the answer is correct
            if (isCorrect) {
                score++;
            }

            // Display feedback
            feedbackElement.textContent = feedbackText;
            feedbackElement.classList.remove('hidden');
            feedbackElement.classList.add('visible');

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

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        triggerConfetti();
                        observer.disconnect();
                    }
                });
            }, { threshold: 1.0 });
            observer.observe(resultsSection);

        } else if (score >= 8) {
            resultsMessage.innerHTML = "Great job! You scored " + score + "/10. You have a strong sense of humor, you're nearly there!";

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        triggerSparkles();
                        observer.disconnect();
                    }
                });
            }, { threshold: 1.0 });
            observer.observe(resultsSection);

        } else if (score >= 6) {
            resultsMessage.innerHTML = "Great job! You scored " + score + "/10. You're funny, but there's still room for improvement.";

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        console.log('Results section is fully visible'); // Debugging log
                        triggerLessConfetti();
                        observer.disconnect();
                    }
                });
            }, { threshold: 1.0 });
            observer.observe(resultsSection);

        } else if (score >= 4) {
            resultsMessage.innerHTML = "You scored " + score + "/10. Not bad, but maybe stick to practicing your jokes on Twitter for now!";

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        console.log('Results section is fully visible'); // Debugging log
                        triggerThumbsUp();
                        observer.disconnect();
                    }
                });
            }, { threshold: 1.0 });
            observer.observe(resultsSection);

        } else {
            resultsMessage.innerHTML = "You scored " + score + "/10. Humor might not be your strong suit, but don't give up!";

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        console.log('Results section is fully visible'); // Debugging log
                        triggerSadEmojis();
                        observer.disconnect();
                    }
                });
            }, { threshold: 1.0 });
            observer.observe(resultsSection);
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

    function triggerSparkles() {
        const defaults = {
            spread: 360,
            ticks: 10,
            gravity: 0,
            decay: 0.98,
            startVelocity: 30,
            shapes: ["star"],
            colors: ["FFE400", "FFBD00", "E89400", "FFCA6C", "FDFFB8"], // Gold and yellow tones
        };
    
        function shoot() {
            confetti({
                ...defaults,
                particleCount: 40,
                scalar: 1.2,
                shapes: ["star"], // Star shapes
            });
    
            confetti({
                ...defaults,
                particleCount: 10,
                scalar: 0.75,
                shapes: ["circle"], // Circle shapes
            });
        }
    
        // Trigger the sparkles multiple times for a dynamic effect
        setTimeout(shoot, 0);
        setTimeout(shoot, 100);
        setTimeout(shoot, 200);
    }

    function triggerLessConfetti() {
        confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 },
            });
    }

    function triggerThumbsUp() {
        const resultsSection = document.getElementById('results');
        const sectionWidth = resultsSection.clientWidth;
        const sectionHeight = resultsSection.clientHeight;
    
        for (let i = 0; i < 60; i++) {
            const thumbsUp = document.createElement('div');
            thumbsUp.textContent = '👍';
            thumbsUp.classList.add('thumbs-up');
    
            const size = Math.random() * 1.5 + 1;
            thumbsUp.style.fontSize = `${size}rem`;
            thumbsUp.style.left = `${Math.random() * (sectionWidth - 50)}px`;
            thumbsUp.style.top = `${Math.random() * (sectionHeight - 50)}px`;
    
            const duration = Math.random() * 4 + 4;
            thumbsUp.style.animation = `fadeUp ${duration}s ease-out forwards`;
    
            resultsSection.appendChild(thumbsUp);
    
            setTimeout(() => {
                thumbsUp.remove();
            }, (duration + 4) * 1000);
        }
    }
    
    function triggerSadEmojis() {
        const resultsSection = document.getElementById('results');
        const sectionWidth = resultsSection.clientWidth;
        const sectionHeight = resultsSection.clientHeight;
    
        for (let i = 0; i < 60; i++) {
            const sadEmoji = document.createElement('div');
            sadEmoji.textContent = '😢';
            sadEmoji.classList.add('sad-emoji');
    
            const size = Math.random() * 1.5 + 1;
            sadEmoji.style.fontSize = `${size}rem`;
            sadEmoji.style.left = `${Math.random() * (sectionWidth - 50)}px`;
            sadEmoji.style.top = `${Math.random() * (sectionHeight - 50)}px`;
    
            const duration = Math.random() * 4 + 4;
            sadEmoji.style.animation = `fadeDown ${duration}s ease-out forwards`;
    
            resultsSection.appendChild(sadEmoji);
    
            setTimeout(() => {
                sadEmoji.remove();
            }, (duration + 4) * 1000);
        }
    }

});