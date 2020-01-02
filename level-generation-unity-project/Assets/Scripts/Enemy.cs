using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour
{
    [SerializeField] private float _speed = 1f;

    void Update()
    {
        Vector3 playerPos = GameManager.GetPlayerPosition();

        // Raycast to player position
        Vector3 fwd = (playerPos - transform.position).normalized;

        RaycastHit hit;
        if (Physics.Raycast(transform.position, fwd, out hit, 100))
        {
            if(hit.collider.GetComponent<Player>() != null)
            {
                // Chase player
                transform.rotation = Quaternion.LookRotation(fwd, Vector3.up);
                transform.position += transform.forward * _speed * Time.deltaTime;
            }
        }
    }

}
