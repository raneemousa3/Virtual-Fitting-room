// Configuration
const API_URL = 'http://localhost:8000';  // Change this in production

// Widget class to handle all functionality
class VirtualFittingRoom {
    constructor() {
        this.modal = null;
        this.init();
    }

    // Initialize the widget
    init() {
        // Create and inject the button
        this.createTryOnButton();
        
        // Create the modal
        this.createModal();
    }

    // Create the "Try It On" button
    createTryOnButton() {
        const button = document.createElement('button');
        button.className = 'vfr-try-on-button';
        button.innerHTML = 'Try It On';
        button.onclick = () => this.openModal();
        
        // Find the add to cart button and insert our button after it
        const addToCartButton = document.querySelector('.shopify-payment-button');
        if (addToCartButton) {
            addToCartButton.parentNode.insertBefore(button, addToCartButton.nextSibling);
        }
    }

    // Create the modal for photo upload
    createModal() {
        const modal = document.createElement('div');
        modal.className = 'vfr-modal';
        modal.innerHTML = `
            <div class="vfr-modal-content">
                <span class="vfr-close">&times;</span>
                <h2>Virtual Try-On</h2>
                <div class="vfr-upload-section">
                    <input type="file" id="vfr-photo-upload" accept="image/*">
                    <div class="vfr-preview"></div>
                </div>
                <div class="vfr-measurements-section" style="display: none;">
                    <h3>Your Measurements</h3>
                    <div class="vfr-measurements"></div>
                    <div class="vfr-fit-rating"></div>
                </div>
                <div class="vfr-loading" style="display: none;">
                    Processing your photo...
                </div>
            </div>
        `;

        // Add event listeners
        modal.querySelector('.vfr-close').onclick = () => this.closeModal();
        modal.querySelector('#vfr-photo-upload').onchange = (e) => this.handlePhotoUpload(e);

        document.body.appendChild(modal);
        this.modal = modal;
    }

    // Open the modal
    openModal() {
        this.modal.style.display = 'block';
    }

    // Close the modal
    closeModal() {
        this.modal.style.display = 'none';
    }

    // Handle photo upload
    async handlePhotoUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Show preview
        const preview = this.modal.querySelector('.vfr-preview');
        preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Preview">`;

        // Show loading
        this.modal.querySelector('.vfr-loading').style.display = 'block';
        this.modal.querySelector('.vfr-measurements-section').style.display = 'none';

        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Send to our API
            const response = await fetch(`${API_URL}/api/v1/measurements/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.displayMeasurements(data.measurements, data.fit_rating);
            } else {
                throw new Error(data.message || 'Failed to process image');
            }
        } catch (error) {
            alert('Error processing image: ' + error.message);
        } finally {
            this.modal.querySelector('.vfr-loading').style.display = 'none';
        }
    }

    // Display measurements and fit rating
    displayMeasurements(measurements, fitRating) {
        const measurementsDiv = this.modal.querySelector('.vfr-measurements');
        const fitRatingDiv = this.modal.querySelector('.vfr-fit-rating');

        // Format measurements
        const measurementsHtml = Object.entries(measurements)
            .map(([key, value]) => `
                <div class="vfr-measurement">
                    <span class="vfr-measurement-label">${key}:</span>
                    <span class="vfr-measurement-value">${value.toFixed(1)} cm</span>
                </div>
            `).join('');

        measurementsDiv.innerHTML = measurementsHtml;
        
        if (fitRating) {
            fitRatingDiv.innerHTML = `
                <h3>Fit Rating</h3>
                <div class="vfr-fit-rating-value">${fitRating}</div>
            `;
        }

        this.modal.querySelector('.vfr-measurements-section').style.display = 'block';
    }
}

// Initialize the widget when the page loads
window.addEventListener('load', () => {
    new VirtualFittingRoom();
}); 