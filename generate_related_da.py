import anthropic
import time
import random
import json
from datetime import datetime
import os
from typing import List, Dict
from tqdm import tqdm

class TravelConfirmationGenerator:
    def __init__(self, api_key: str):
        """
        Initialize travel confirmation text generator

        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.generated_texts = []

    def generate_batch_prompt(self, batch_size: int = 50) -> str:
        """
        Generate batch prompt for text generation

        Args:
            batch_size: Number of texts to generate in one batch

        Returns:
            Complete prompt string
        """
        prompt = f"""# Unified Travel Booking Confirmation Generator

Generate exactly {batch_size} diverse travel booking confirmation texts that simulate OCR-scanned reservation documents from various booking platforms across three categories: hotel bookings, flight bookings, and car rentals. Each confirmation should be on a single line.

## Category 1: Hotel Booking Confirmations

**Required Information:**
- Confirmation/Booking reference number
- Guest name(s)
- Hotel name and complete address
- Check-in and check-out dates
- Room type/category
- Number of guests/occupancy
- Total nights stayed
- Final price amount

**Format Variations:**
- Standard confirmation format with line breaks
- Compact single-line text format
- Email receipt style with headers
- Mobile app notification style
- Property management system printout
- Third-party booking site confirmation

## Category 2: Flight Booking Confirmations

**Required Information:**
- Booking reference/PNR code
- Passenger name(s)
- Airline name and flight number
- Departure and arrival airports (codes and full names)
- Departure and arrival dates/times
- Seat number(s) or class
- Number of passengers
- Ticket price/total cost

**Format Variations:**
- Traditional e-ticket format
- Mobile boarding pass style
- Airline app confirmation
- Travel agency booking format
- Multi-segment itinerary layout
- Compact SMS confirmation style

## Category 3: Car Rental Confirmations

**Required Information:**
- Reservation number/confirmation code
- Renter name (primary driver)
- Rental company name
- Pick-up and drop-off locations
- Pick-up and drop-off dates/times
- Vehicle type/category/model
- Rental duration (days/hours)
- Total rental cost

**Format Variations:**
- Standard rental agreement format
- Online booking confirmation style
- Mobile app reservation format
- Airport counter receipt style
- Compact voucher format
- Email confirmation layout

## Generation Instructions

**For Each Confirmation:**
- Use realistic business names, locations, dates, and pricing
- Vary terminology across documents (Guest/Traveler/Customer, Hotel/Property/Resort, Vehicle/Car/Auto, etc.)
- Include different date and time formats (MM/DD/YYYY, DD-MM-YYYY, 12hr/24hr)
- Create both domestic and international scenarios where applicable
- Include various price ranges and booking durations
- Mix compact single-line formats with structured multi-line layouts
- Include both formal business documents and casual confirmation styles
- Simulate different OCR quality levels (clean vs. slightly distorted text)
- Vary font styles, spacing, and alignment patterns
- Include abbreviated and full-form text variations
- Use actual airport codes, hotel chains, and rental companies
- Include realistic confirmation codes (6-8 alphanumeric characters)
- Generate appropriate pricing for different service levels

**Output Format:**
- Each confirmation should be exactly one line
- Put each confirmation on a separate line
- No numbering or bullets
- Mix categories randomly rather than grouping
- Generate exactly {batch_size} confirmations

**Example Output Format:**
CONFIRMATION: HTL789234 | GUEST: Sarah Johnson | HILTON GARDEN INN DOWNTOWN | 123 MAIN ST, CHICAGO IL | CHECK-IN: 03/15/2024 14:00 | CHECK-OUT: 03/18/2024 11:00 | ROOM: KING DELUXE | GUESTS: 2 | NIGHTS: 3 | TOTAL: $427.50
FLIGHT BOOKING PNR: ABC123 | PASSENGER: MIKE CHEN | UNITED AIRLINES UA1245 | LAX-JFK | DEPART: 2024-04-22 08:30 | ARRIVE: 2024-04-22 17:15 | SEAT: 14A | ECONOMY | FARE: $389.00
RENTAL CONFIRMATION R567890 | DRIVER: Jennifer Smith | HERTZ | PICKUP: Orlando Airport Terminal B | DROPOFF: Same Location | 05/10/2024 10:00 - 05/15/2024 10:00 | VEHICLE: Toyota Camry or Similar | 5 DAYS | $234.75 TOTAL

IMPORTANT: Generate exactly {batch_size} confirmation texts, one per line. Do not include any explanations, headers, or extra text. Only output the confirmation texts themselves:"""

        return prompt

    def generate_texts_batch(self, batch_size: int = 50) -> List[str]:
        """
        Generate a batch of texts

        Args:
            batch_size: Number of texts to generate in one batch

        Returns:
            List of generated texts
        """
        try:
            prompt = self.generate_batch_prompt(batch_size)

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,
                temperature=0.8,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse response, extract each line as separate confirmation text
            lines = response_text.strip().split('\n')
            confirmations = []

            for line in lines:
                line = line.strip()
                # Filter out empty lines and lines that are obviously not confirmation texts
                if line and not line.startswith('#') and not line.startswith('**') and len(line) > 50:
                    confirmations.append(line)

            return confirmations[:batch_size]  # Ensure not exceeding requested count

        except Exception as e:
            print(f"Error generating texts: {e}")
            return []

    def generate_all_texts(self, total_count: int = 30000, batch_size: int = 50, output_file: str = "travel_confirmations.txt"):
        """
        Generate all texts and save to file

        Args:
            total_count: Total number of texts to generate
            batch_size: Number of texts per batch
            output_file: Output filename
        """
        print(f"Starting generation of {total_count} travel booking confirmation texts...")
        print(f"Generating {batch_size} per batch, total {total_count // batch_size} batches needed")

        all_texts = []
        total_batches = (total_count + batch_size - 1) // batch_size

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                with tqdm(total=total_count, desc="Generating texts", unit="texts") as pbar:
                    while len(all_texts) < total_count:
                        remaining = total_count - len(all_texts)
                        current_batch_size = min(batch_size, remaining)

                        batch_texts = self.generate_texts_batch(current_batch_size)

                        if batch_texts:
                            # Write to file
                            for text in batch_texts:
                                f.write(text + '\n')
                                all_texts.append(text)

                            f.flush()  # Force write to disk
                            pbar.update(len(batch_texts))
                        else:
                            print("Current batch failed, waiting 5 seconds before retry...")
                            time.sleep(5)
                            continue

                        # Add delay between batches to avoid API limits
                        if len(all_texts) < total_count:
                            time.sleep(2)  # Wait 2 seconds

        except KeyboardInterrupt:
            print(f"\nGeneration interrupted, saved {len(all_texts)} texts to {output_file}")
        except Exception as e:
            print(f"Error during generation: {e}")

        print(f"Generation complete! Generated {len(all_texts)} texts, saved to {output_file}")
        return all_texts

    def load_existing_texts(self, filename: str) -> List[str]:
        """
        Load existing texts from file

        Args:
            filename: Filename

        Returns:
            List of texts
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            return []

    def resume_generation(self, output_file: str = "travel_confirmations.txt", target_count: int = 30000, batch_size: int = 50):
        """
        Resume generation process (if previously interrupted)

        Args:
            output_file: Output filename
            target_count: Target total count
            batch_size: Batch size
        """
        existing_texts = self.load_existing_texts(output_file)
        current_count = len(existing_texts)

        if current_count >= target_count:
            print(f"File {output_file} already has {current_count} texts, target reached")
            return existing_texts

        print(f"Resuming from {current_count} texts, target {target_count} texts")
        remaining = target_count - current_count

        # Continue generating remaining texts
        new_texts = []
        with open(output_file, 'a', encoding='utf-8') as f:
            with tqdm(total=remaining, desc="Resuming generation", unit="texts") as pbar:
                while len(new_texts) < remaining:
                    batch_remaining = min(batch_size, remaining - len(new_texts))

                    batch_texts = self.generate_texts_batch(batch_remaining)

                    if batch_texts:
                        for text in batch_texts:
                            f.write(text + '\n')
                            new_texts.append(text)

                        f.flush()
                        pbar.update(len(batch_texts))

                    if len(new_texts) < remaining:
                        time.sleep(2)

        return existing_texts + new_texts

def main():
    """
    Main function - run text generation
    """
    # Get API key from environment variable or direct input
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable or provide API key directly")
        print("Example: export ANTHROPIC_API_KEY='your-api-key-here'")
        api_key = input("Please enter your Anthropic API key: ").strip()

    if not api_key:
        print("No API key provided, exiting")
        return

    # Create generator
    generator = TravelConfirmationGenerator(api_key)

    # Set parameters
    total_texts = 30000  # Generate 100 texts for testing
    batch_size = 100      # 50 texts per batch
    output_file = "travel_confirmations.txt"

    print("=" * 60)
    print("Travel Booking Confirmation Text Generator")
    print("=" * 60)
    print(f"Target count: {total_texts:,} texts")
    print(f"Batch size: {batch_size} texts")
    print(f"Output file: {output_file}")
    print("=" * 60)

    # Check if need to resume generation
    choice = input("Resume previous generation progress? (y/n): ").lower()

    if choice == 'y':
        texts = generator.resume_generation(output_file, total_texts, batch_size)
    else:
        # If file exists, ask whether to overwrite
        if os.path.exists(output_file):
            overwrite = input(f"File {output_file} exists, overwrite? (y/n): ").lower()
            if overwrite != 'y':
                print("Exiting")
                return

        texts = generator.generate_all_texts(total_texts, batch_size, output_file)

    print(f"\nGeneration complete! File saved at: {output_file}")
    print(f"Total generated: {len(texts):,} travel confirmation texts")

if __name__ == "__main__":
    main()
