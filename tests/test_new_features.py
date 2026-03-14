"""Test new API features."""
import sys
sys.path.insert(0, 'src')
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    # Test version
    print('=== Version ===')
    v = client.get_version()
    print(f'Version: {v.version}')
    
    # Test modes
    print('\n=== Modes ===')
    for m in client.get_modes():
        desc = m.description[:50] if m.description else ''
        print(f'  {m.name} ({m.kind}): {desc}...')
    
    # Test agents
    print('\n=== Agents ===')
    for a in client.get_agents():
        print(f'  {a.slug}: {a.name}')
    
    # Test copilot models
    print('\n=== Models (first 5) ===')
    for m in client.get_copilot_models()[:5]:
        print(f'  {m.id}: {m.name} (premium={m.is_premium})')
    
    # Test templates
    print('\n=== Templates ===')
    for t in client.get_templates()[:5]:
        print(f'  /{t.id}: {t.short_description}')
    
    # Test conversation with mode
    print('\n=== Chat with Agent mode ===')
    conv = client.create_conversation(mode='Agent')
    print(f'Conversation mode: {conv.mode}')
    resp = client.send_message(conv, 'Say hello in 3 words')
    print(f'Response: {resp.content}')
    print(f'Messages in conv: {len(conv.messages)}')
    for i, msg in enumerate(conv.messages):
        content = msg.content[:50] if len(msg.content) > 50 else msg.content
        print(f'  {i+1}. {msg.role}: {content}...')

