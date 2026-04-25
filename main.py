from graph import ask_question

def main():
    """
    Simple chat interface for the RAG Customer Support Bot
    """
    print("="*60)
    print("TECHFLOW SOLUTIONS - CUSTOMER SUPPORT BOT")
    print("="*60)
    print("\nHello! I'm your AI assistant. I can help you with:")
    print("  • Account management (password reset, billing)")
    print("  • Product features (AI assistant, integrations)")
    print("  • Technical support (bugs, errors)")
    print("  • Pricing and subscriptions")
    print("  • Data security questions")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("="*60)
    
    # Chat loop
    while True:
        # Get user input
        print("\n" + "─"*60)
        question = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\nThank you for contacting TechFlow Support! Have a great day!")
            break
        
        # Skip empty questions
        if not question:
            print("Please ask a question!")
            continue
        
        # Ask the bot
        try:
            result = ask_question(question)
            
            # Print the answer in a nice format
            print("\n" + "─"*60)
            print("Bot:")
            print("─"*60)
            print(result["answer"])
            
            # Show if it needed human help
            if result["needs_human"]:
                print("\nThis question has been escalated to a human agent.")
                print("   They will get back to you shortly!")
            
        except Exception as e:
            print(f"\nOops! Something went wrong: {e}")
            print("   Please try asking your question differently.")


if __name__ == "__main__":
    main()