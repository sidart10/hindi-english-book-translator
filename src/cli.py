#!/usr/bin/env python3
"""
Command Line Interface for Hindi-English Book Translation System
Provides easy-to-use commands for translating books
"""

import click
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
import json

# Import our components
from main_controller import BookTranslationController


def validate_input_file(ctx, param, value):
    """Validate that input file exists and is supported"""
    if not value:
        return value
    
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Input file '{value}' does not exist")
    
    supported_extensions = ['.pdf', '.epub', '.txt', '.docx']
    if path.suffix.lower() not in supported_extensions:
        raise click.BadParameter(
            f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )
    
    return str(path.absolute())


def validate_config_file(ctx, param, value):
    """Validate that config file exists if specified"""
    if not value:
        return value
    
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"Config file '{value}' does not exist")
    
    return str(path.absolute())


@click.command()
@click.option(
    '--input', '-i',
    required=True,
    callback=validate_input_file,
    help='Input book file (PDF/EPUB/TXT/DOCX)'
)
@click.option(
    '--output', '-o',
    required=True,
    help='Output DOCX file path'
)
@click.option(
    '--config', '-c',
    default='config.json',
    callback=validate_config_file,
    help='Configuration file path (default: config.json)'
)
@click.option(
    '--project-id', '-p',
    envvar='GOOGLE_CLOUD_PROJECT',
    help='Google Cloud Project ID (can also be set via GOOGLE_CLOUD_PROJECT env var)'
)
@click.option(
    '--service-account', '-s',
    envvar='GOOGLE_APPLICATION_CREDENTIALS',
    help='Path to Google Cloud service account JSON key file'
)
@click.option(
    '--batch-size', '-b',
    type=int,
    help='Number of sentences to process in each batch (overrides config)'
)
@click.option(
    '--monthly-budget',
    type=float,
    help='Monthly budget in USD (overrides config)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Analyze the document without translating (shows estimated cost)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed progress information'
)
def translate_book(input, output, config, project_id, service_account, 
                  batch_size, monthly_budget, dry_run, verbose):
    """
    Translate a Hindi book to English using Google Cloud Translation API
    
    Examples:
    
        # Basic usage with environment variables already set
        book-translator --input book.pdf --output translation.docx
        
        # Specify Google Cloud credentials
        book-translator -i book.pdf -o translation.docx \\
            --project-id my-project \\
            --service-account /path/to/key.json
        
        # Dry run to estimate cost
        book-translator -i book.pdf -o translation.docx --dry-run
        
        # Custom batch size and budget
        book-translator -i book.pdf -o translation.docx \\
            --batch-size 100 --monthly-budget 500
    """
    
    # Display banner
    click.echo("=" * 70)
    click.echo("📚 Hindi-English Book Translation System")
    click.echo("=" * 70)
    
    # Validate and set up Google Cloud credentials
    if service_account:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account
        click.echo(f"✅ Using service account: {service_account}")
    elif not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        # Try to find service-account.json in current directory
        default_path = Path('service-account.json')
        if default_path.exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(default_path.absolute())
            click.echo(f"✅ Found service account: {default_path}")
        else:
            click.echo(
                "❌ Error: No Google Cloud credentials found.\n"
                "   Please provide --service-account or set GOOGLE_APPLICATION_CREDENTIALS",
                err=True
            )
            sys.exit(1)
    
    # Set project ID if provided
    if project_id:
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        click.echo(f"✅ Using project ID: {project_id}")
    
    # Load and potentially override configuration
    config_data = {}
    if Path(config).exists():
        with open(config, 'r') as f:
            config_data = json.load(f)
        click.echo(f"✅ Loaded configuration from: {config}")
    
    # Apply CLI overrides
    if batch_size:
        if 'processing' not in config_data:
            config_data['processing'] = {}
        config_data['processing']['batch_size'] = batch_size
        click.echo(f"✅ Batch size set to: {batch_size}")
    
    if monthly_budget:
        if 'cost' not in config_data:
            config_data['cost'] = {}
        config_data['cost']['monthly_budget'] = monthly_budget
        click.echo(f"✅ Monthly budget set to: ${monthly_budget}")
    
    # Save temporary config if overrides were applied
    temp_config = None
    if batch_size or monthly_budget:
        temp_config = "temp_config.json"
        with open(temp_config, 'w') as f:
            json.dump(config_data, f, indent=2)
        config = temp_config
    
    click.echo("")
    
    # Display input information
    input_path = Path(input)
    click.echo(f"📖 Input file: {input_path.name}")
    click.echo(f"   Size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
    click.echo(f"   Format: {input_path.suffix.upper()}")
    
    if dry_run:
        click.echo("\n🔍 DRY RUN MODE - Analyzing document...")
        
        # Import document processor for analysis
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Count sentences and estimate cost
        sentences = list(processor.process_document(input))
        total_chars = sum(len(s.text) if hasattr(s, 'text') else len(s.get('text', '')) 
                         for s in sentences)
        estimated_cost = (total_chars / 1000) * 0.20
        
        click.echo(f"\n📊 Document Analysis:")
        click.echo(f"   Total sentences: {len(sentences):,}")
        click.echo(f"   Total characters: {total_chars:,}")
        click.echo(f"   Estimated cost: ${estimated_cost:.2f}")
        click.echo(f"   Cost per page: ${estimated_cost / max(1, len(sentences) / 20):.2f}")
        
        # Check budget
        budget = config_data.get('cost', {}).get('monthly_budget', 1000)
        if estimated_cost > budget * 0.7:
            click.echo(f"\n⚠️  Warning: Estimated cost exceeds 70% of monthly budget (${budget})")
        
        click.echo("\n✅ Dry run complete. No translation was performed.")
        
        # Clean up temp config
        if temp_config and Path(temp_config).exists():
            os.remove(temp_config)
        
        return
    
    # Run the actual translation
    try:
        click.echo(f"\n🚀 Starting translation...")
        click.echo(f"   Output will be saved to: {output}")
        
        # Initialize controller
        controller = BookTranslationController(config)
        
        # Run translation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            controller.translate_book(input, output)
        )
        
        # Display results
        click.echo("\n" + "=" * 70)
        click.echo("✅ TRANSLATION COMPLETE!")
        click.echo("=" * 70)
        
        stats = results['statistics']
        click.echo(f"\n📊 Translation Statistics:")
        click.echo(f"   Total sentences: {stats['total_sentences']:,}")
        click.echo(f"   Translated sentences: {stats['translated_sentences']:,}")
        click.echo(f"   Total characters: {stats['total_characters']:,}")
        click.echo(f"   Total cost: ${stats['total_cost']:.2f}")
        
        if stats.get('quality_warnings'):
            click.echo(f"\n⚠️  Quality warnings: {len(stats['quality_warnings'])}")
            if verbose:
                for warning in stats['quality_warnings'][:5]:
                    click.echo(f"   - {warning}")
        
        if stats.get('errors'):
            click.echo(f"\n❌ Errors encountered: {len(stats['errors'])}")
            for error in stats['errors']:
                click.echo(f"   - {error}")
        
        # Show output file info
        output_path = Path(output)
        if output_path.exists():
            click.echo(f"\n📄 Output file created:")
            click.echo(f"   Path: {output_path.absolute()}")
            click.echo(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Cost summary
        click.echo(f"\n💰 Cost Summary:")
        click.echo(f"   Translation cost: ${stats['total_cost']:.2f}")
        click.echo(f"   Per sentence: ${stats['total_cost'] / max(1, stats['total_sentences']):.4f}")
        
        # Check budget status
        budget = config_data.get('cost', {}).get('monthly_budget', 1000)
        budget_used = (stats['total_cost'] / budget) * 100
        click.echo(f"   Budget used: {budget_used:.1f}% of ${budget}")
        
        if budget_used > 70:
            click.echo("\n⚠️  Warning: Over 70% of monthly budget used!")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up temporary config
        if temp_config and Path(temp_config).exists():
            os.remove(temp_config)
    
    click.echo("\n✨ Translation completed successfully!")


# Entry point for the package
def main():
    """Main entry point for the CLI"""
    translate_book()


if __name__ == '__main__':
    main() 