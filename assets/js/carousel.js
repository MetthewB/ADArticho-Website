// Image Carousel for Banner
$(document).ready(function() {
    var imageElement = $('#banner-image');
    var images = [
        "images/patate.jpg",
        "images/cactus.png",
        "images/alien.jpg",
        "images/dealer.jpg",
        "images/magic.jpg",
        "images/snowman.jpg",
        "images/swimmer.jpg"
    ];
    var currentIndex = 0;

    setInterval(function() {
        // Add the hidden class for fade-out effect
        imageElement.addClass('hidden');

        // Wait for the fade-out transition to complete
        setTimeout(function() {
            // Update the image source
            currentIndex = (currentIndex + 1) % images.length;
            imageElement.attr('src', images[currentIndex]);

            // Remove the hidden class for fade-in effect
            imageElement.removeClass('hidden');
        }, 500); // Match the CSS transition duration
    }, 5000); // Change image every 5 seconds
});