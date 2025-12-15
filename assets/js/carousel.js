// Image Carousel for Banner
$(document).ready(function() {
    var imageElement = $('#banner-image');
    var captionElement = $('#banner-caption'); // Add a reference to the caption element
    var images = [
        { src: "images/patate.jpg", caption: "Winning Caption: Congratulations. We've found millions of compatible donors." },
        { src: "images/cactus.png", caption: "Winning Caption: You know, it's gonna itch like hell when it grows back." },
        { src: "images/alien.jpg", caption: "Winning Caption: Not as surprising as watching you pick up his poo." },
        { src: "images/dealer.jpg", caption: "Winning Caption: When I said “wait till you see these puppies,” what did you think I meant?" },
        { src: "images/magic.jpg", caption: "Winning Caption: Be grateful you're not in business class. They have a mime." },
        { src: "images/snowman.jpg", caption: "Winning Caption: Upset? I’m beside myself!" },
        { src: "images/swimmer.jpg", caption: "Winning Caption: Are you very satisfied, somewhat satisfied, or not at all satisfied with your pool experience?" },
        { src: "images/seagull.jpg", caption: "Winning Caption: And I'm the dirty one?" },
        { src: "images/air.jpg", caption: "Winning Caption: I absolutely love what you’ve done with your air." },
        { src: "images/wine.jpg", caption: "Winning Caption: You can leave your label on." }
    ];
    var currentIndex = 0;

    setInterval(function() {
        // Add the hidden class for fade-out effect
        imageElement.addClass('hidden');
        captionElement.addClass('hidden'); // Fade out the caption as well

        // Wait for the fade-out transition to complete
        setTimeout(function() {
            // Update the image source and caption
            currentIndex = (currentIndex + 1) % images.length;
            imageElement.attr('src', images[currentIndex].src);
            captionElement.text(images[currentIndex].caption);

            // Remove the hidden class for fade-in effect
            imageElement.removeClass('hidden');
            captionElement.removeClass('hidden');
        }, 500); // Match the CSS transition duration
    }, 5000); // Change image every 5 seconds
});