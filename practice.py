for epoch in range(10):
    running_loss = 0.0
    
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        outputs = net(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")