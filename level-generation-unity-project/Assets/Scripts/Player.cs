using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : MonoBehaviour
{
    [SerializeField] private float _speed = 1f;

    private void Start()
    {
        GameManager.RegisterPlayer(this);
    }

    // Update is called once per frame
    void Update()
    {
        float dirX = 0f;
        float dirY = 0f;
        if(Input.GetKey(KeyCode.RightArrow))
            dirX++;
        else if(Input.GetKey(KeyCode.LeftArrow))
            dirX--;
        if(Input.GetKey(KeyCode.UpArrow))
            dirY++;
        else if(Input.GetKey(KeyCode.DownArrow))
            dirY--;

        Vector3 delta = (transform.right * dirX + transform.forward * dirY).normalized * Time.deltaTime;
        transform.position += delta;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.GetComponent<Enemy>() != null)
        {
            Destroy(gameObject);
        }
    }
}
