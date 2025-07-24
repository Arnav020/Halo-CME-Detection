document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  const fileInput = document.querySelector("input[type='file']");
  const submitButton = document.querySelector("button");

  form.addEventListener("submit", function (event) {
    let valid = true;

    // Reset styling
    if (fileInput) {
      fileInput.style.border = "1px solid #ccc";
    }

    // Validate file input
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      if (fileInput) {
        fileInput.style.border = "2px solid red";
      }
      alert("⚠️ Please upload a valid CSV file before submitting.");
      valid = false;
    }

    if (!valid) {
      event.preventDefault();
    } else {
      // Optional: Show loading state on button
      submitButton.disabled = true;
      submitButton.textContent = "Predicting...";
    }
  });
});


